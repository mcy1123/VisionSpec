"""Explicit three-card continuation with four logical QA slots and slot RNG streams.

Only frozen Target tensors cross processes. Student context projections and every
anchor backward are rebuilt on rank0 by runtime.score in the original order.
"""
import datetime
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from shared import common, training_resume
from shared.master_optim import MasterWeightAdamW
from . import adapters, runtime
from .native_contracts import bind_trace, denominator_record, global_denominator
from .rollout_parallel import (head_digest, parameter_versions, check_versions,
    validate_envelope, send_trace, receive_trace, budget_event, broadcast_head)
from .trainer import move_trace
from .cli import STATUS as ORIGINAL_STATUS
from .rollout_parallel import TRACE_FIELDS, tensors
from .native_contracts import validate_trace

WORLD = 3
SLOTS = 4
OPD_MODE = "pure_block"
SLOT_MAPPING = {"0": [0, 3], "1": [1], "2": [2]}
STATUS = dict(ORIGINAL_STATUS, implementation="pure block-native; three ranks, four logical slots; ordered rank0 scoring")


def identity(args):
    ignored = {'out','gpu','resume_from','resume_update','stop_after','dry_run',
               'no_update','stop_file','save_every','deadline_epoch','experiment_budget','experiment_case',
               'experiment_entry'}
    config = {k:v for k,v in vars(args).items() if k not in ignored}
    if not config.get('allow_partial_final'):
        config.pop('allow_partial_final', None)
    initial = dict(checkpoint_model_sha256=training_resume.file_hash(
        Path(args.init_from)/'model.safetensors'), optimizer_step=0)
    config.update(recipe=STATUS, opd_mode=OPD_MODE, world_size=WORLD,
                  slot_map=SLOT_MAPPING, initial_head_version=initial)
    return dict(opd_mode=OPD_MODE, world_size=WORLD, slot_map=SLOT_MAPPING,
        initial_head_version=initial,
        schedule_sha256=training_resume.file_hash(args.schedule),
        pool_sha256=training_resume.file_hash(args.pool),
        model_config_sha256=training_resume.file_hash(Path(args.init_from)/'config.json'),
        config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())


def validate_envelope(trace, identity, policy, metadata):
    if (set(trace) != TRACE_FIELDS or trace.get('teacher_verify_impl') != 'pure_block_native'
        or trace.get('training_identity') != dict(metadata, head_version=policy)
        or metadata.get('opd_mode') != 'pure_block' or metadata.get('world_size') != WORLD
        or metadata.get('slot_map') != SLOT_MAPPING):
        raise ValueError('native-only trace/mode/topology/Head identity required')
    validate_trace(trace, identity)
    if trace['policy_identity'] != policy:
        raise ValueError('stale or cross-update Head trace')
    check_versions([policy])
    if identity['checkpoint_sha256'] != policy['head_sha256']:
        raise ValueError('trace checkpoint and policy Head disagree')
    for t in tensors(trace):
        if t.requires_grad or t.grad_fn is not None or t.device.type != 'cpu':
            raise ValueError('wire trace must contain detached CPU rollout tensors only')
    if trace['hidden'].dtype != torch.float32 or trace['fused'].dtype != torch.float32:
        raise ValueError('frozen fp32 Target state precision changed')
    if (trace['ids'].dtype != torch.long
        or trace['hidden'].shape[1] != trace['ids'].numel()
        or trace['fused'].shape[1] != trace['ids'].numel()
        or trace['final_cache_length'] != trace['ids'].numel()):
        raise ValueError('trace token/Target context alignment changed')


def validate_checkpoint(state, metadata, *, smoke=False, requested_step=None):
    """Identity protection only; resume is not part of this smoke validation."""
    if (state.get('resume_schema') != 2 or not state.get('block_native')
        or state.get('opd_mode') != OPD_MODE or state.get('resume_identity') != metadata):
        raise ValueError('pure-block checkpoint identity required; cross-mode resume is forbidden')
    if state.get('smoke', False) != smoke or state.get('diagnostic_only', False) != smoke:
        raise ValueError('smoke/diagnostic checkpoints cannot become normal training states')
    if state['update_step'] != state['data_cursor'] or state.get('update') != state['data_cursor']:
        raise ValueError('checkpoint cursor/update mismatch')
    if requested_step is not None and requested_step != state['data_cursor']:
        raise ValueError('requested resume step mismatch')
    p = state.get('parallel_state')
    if (not p or p.get('world_size') != WORLD or p.get('slot_mapping') != SLOT_MAPPING
        or p.get('implementation') != 'pure_block_three_card_four_slot_v1'
        or len(p.get('per_slot_rng', [])) != SLOTS
        or p.get('schedule_cursor') != state['data_cursor']):
        raise ValueError('checkpoint topology/slot/RNG/cursor mismatch; no cross-world resume')
    version = dict(head_sha256=p['head_sha256'], optimizer_step=p['optimizer_step'],
                   schedule_cursor=p['schedule_cursor'])
    if state.get('training_identity') != dict(metadata, head_version=version):
        raise ValueError('checkpoint Head-version identity mismatch')
    return state['data_cursor']


def local_rng(device):
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                python=random.getstate(), torch_cuda=[torch.cuda.get_rng_state(device)])


def restore_local_rng(state, device):
    training_resume.restore_rng(dict(state, torch_cuda=[]))
    if len(state['torch_cuda']) != 1:
        raise ValueError('checkpoint must contain exactly one local CUDA RNG state')
    torch.cuda.set_rng_state(state['torch_cuda'][0], device)


class ParallelRollout:
    def __init__(self, head, metadata, device):
        self.head, self.metadata, self.device = head, metadata, device
        self.rank = dist.get_rank()
        self.slot_rng = {}
        self.nccl = dist.new_group(backend='nccl', timeout=datetime.timedelta(minutes=20))
        # Initialize communicator before restoring the checkpoint RNG.
        dist.broadcast(torch.zeros(1, device=device), 0, group=self.nccl)
        self.versions = parameter_versions(head)
        self.head_sha = head_digest(head)

    def gather(self, value):
        values = [None] * WORLD
        dist.all_gather_object(values, value)
        return values

    def announce(self, value):
        box = [value]
        dist.broadcast_object_list(box, 0)
        return box[0]

    def phase_check(self, error):
        errors = self.gather(error)
        if any(errors):
            raise RuntimeError('parallel update aborted before commit: ' + repr(errors))

    def barrier(self, optimizer_step, cursor):
        self.phase_check(None if parameter_versions(self.head) == self.versions
                         else 'Head mutated outside version barrier')
        policy = dict(head_sha256=self.head_sha, optimizer_step=optimizer_step,
                      schedule_cursor=cursor)
        check_versions(self.gather(policy))
        dist.barrier()
        return policy

    def collect(self, target, processor, rows, items, args, optimizer_step, cursor):
        if [i['slot'] for i in items] != list(range(len(items))) or not 1 <= len(items) <= SLOTS:
            raise ValueError('three-card rollout requires canonical logical slot0..3 exactly once')
        policy = self.barrier(optimizer_step, cursor)
        identities = []
        for slot, item in enumerate(items):
            row = rows.loc[item['task']].to_dict()
            prompt_sha = hashlib.sha256(row['prompt'].encode()).hexdigest()
            if item.get('prompt_sha256', prompt_sha) != prompt_sha:
                raise ValueError('schedule/pool prompt mismatch')
            identities.append(dict(checkpoint_sha256=self.head_sha, task=item['task'],
                slot=slot, prompt_sha256=prompt_sha,
                config_sha256=self.metadata['config_sha256']))
        error = None
        traces, local_records = {}, {}
        try:
            for slot in SLOT_MAPPING[str(self.rank)]:
                if slot < len(items):
                    restore_local_rng(self.slot_rng[slot], self.device)
                    traces[slot], local_records[slot] = self.collect_slot(
                        target, processor, rows, items, args, identities, policy, slot)
                    self.slot_rng[slot] = local_rng(self.device)
            # Ordered backward uses the old logical rank0 RNG stream, not slot3.
            if self.rank == 0:
                restore_local_rng(self.slot_rng[0], self.device)
        except Exception as exc:
            error = repr(exc)
        self.phase_check(error)
        all_records = {slot: record for bundle in self.gather(local_records)
                       for slot, record in bundle.items()}
        records = [all_records[slot] for slot in range(len(items))]
        denominator = global_denominator(records, [i['task'] for i in items])
        if self.announce(denominator.hex() if self.rank == 0 else None) != denominator.hex():
            raise RuntimeError('global denominator binary64 disagreement')
        wire_bytes = 0
        if self.rank:
            for slot in SLOT_MAPPING[str(self.rank)]:
                if slot in traces:
                    wire_bytes += send_trace(traces[slot])
        else:
            for source in range(1, WORLD):
                for slot in SLOT_MAPPING[str(source)]:
                    if slot < len(items):
                        traces[slot], n = receive_trace(source)
                        wire_bytes += n
        error = None
        if self.rank == 0:
            try:
                for slot in range(len(items)):
                    validate_envelope(traces[slot], identities[slot], policy, self.metadata)
                received = [denominator_record(traces[s], s, items[s]['task']) for s in range(len(items))]
                if received != records or sum(runtime.denominator(traces[s]) for s in range(len(items))) != denominator:
                    raise ValueError('received trace denominator differs')
            except Exception as exc:
                error = repr(exc)
        self.phase_check(error)
        self.barrier(optimizer_step, cursor)
        return traces, denominator, policy, records, wire_bytes

    def collect_slot(self, target, processor, rows, items, args, identities, policy, slot):
        inputs = adapters.encode(processor, rows.loc[items[slot]['task']].to_dict(),
            args.modality, max_pixels=args.max_pixels, time_mode=args.time_mode)
        trace = adapters.collect(target, self.head, inputs,
            max_new_tokens=args.max_new_tokens, max_seq_len=args.max_seq_len,
            teacher_verify_impl=args.teacher_verify_impl, batch_teacher_ids=False,
            preallocate_context=False)
        trace = move_trace(trace, 'cpu')
        bind_trace(trace, identities[slot])
        trace['policy_identity'] = policy
        trace['training_identity'] = dict(self.metadata, head_version=policy)
        validate_envelope(trace, identities[slot], policy, self.metadata)
        record = denominator_record(trace, slot, items[slot]['task'])
        return trace, record

    def commit(self, optimizer_step, cursor, changed):
        if changed:
            broadcast_head(self.head, self.nccl)
        self.head_sha = head_digest(self.head)
        self.versions = parameter_versions(self.head)
        self.barrier(optimizer_step, cursor)


def run(args, allow_fresh_start=False):
    import pandas as pd
    if (int(os.environ.get('WORLD_SIZE', 1)) != WORLD or torch.cuda.device_count() < WORLD
        or args.parallel_rollout != WORLD or args.opd_mode != OPD_MODE
        or args.qa_per_update != SLOTS or args.target_dtype != 'float32'
        or args.master_weights != 'cpu' or args.teacher_verify_impl != 'pure_block_native'):
        raise ValueError('three visible GPUs and torchrun --nproc_per_node=3 required; no fallback')
    training_resume.validate_output(args.out)
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = f'cuda:{local}'
    dist.init_process_group('gloo', timeout=datetime.timedelta(minutes=20))
    rank = dist.get_rank()
    try:
        out = Path(args.out)
        if rank == 0:
            if out.exists() and any(out.iterdir()):
                raise ValueError('new output must be empty; resume into a new output directory')
            out.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        metadata = identity(args)
        rows = pd.read_parquet(args.pool).set_index('task', verify_integrity=True)
        schedule = {}
        for line in Path(args.schedule).read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                item['task'] = item.get('task') or f"{item['video_id']}|{item['question_id']}"
                schedule.setdefault(int(item['update']), []).append(item)
        if sorted(schedule) != list(range(args.updates)):
            raise ValueError('schedule must cover exactly all requested updates')
        for cursor, items in schedule.items():
            items.sort(key=lambda x: x['slot'])
            partial = args.allow_partial_final and (cursor + 1) % args.epoch_updates == 0 and len(items) == 2
            if (len(items) != SLOTS and not partial) or [x['slot'] for x in items] != list(range(len(items))):
                raise ValueError('invalid slot count/order; partial final requires explicit opt-in')
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
        target, processor = common.load_target(device=device, dtype=torch.float32)
        target.eval().requires_grad_(False)
        head = common.load_head(args.init_from, device=device, dtype=torch.bfloat16).eval()
        head.requires_grad_(rank == 0)
        optimizer = (MasterWeightAdamW(head, lr=args.lr, betas=common.BETAS,
            weight_decay=common.WEIGHT_DECAY, where='cpu', clip=args.clip) if rank == 0 else None)
        executor = ParallelRollout(head, metadata, device)
        start, optimizer_step = 0, 0
        if args.resume_from:
            state = torch.load(Path(args.resume_from)/'training_state.pt', map_location='cpu', weights_only=False)
            start = validate_checkpoint(state, metadata, smoke=args.smoke, requested_step=args.resume_update)
            head.load_state_dict(state['head'])
            if optimizer:
                optimizer.load_state_dict(state['master_optimizer'])
            p = state['parallel_state']
            optimizer_step = p['optimizer_step']
            if head_digest(head) != p['head_sha256']:
                raise ValueError('checkpoint Head/version hash mismatch')
            executor.slot_rng = {slot: p['per_slot_rng'][slot] for slot in SLOT_MAPPING[str(rank)]}
            restore_local_rng(executor.slot_rng[SLOT_MAPPING[str(rank)][0]], device)
            del state
            if optimizer and {float(v['step']) for v in optimizer._opt.state.values()} != {float(optimizer_step)}:
                raise ValueError('Adam step differs from checkpoint Head version')
        else:
            if not allow_fresh_start:
                raise ValueError('three-card continuation requires an explicitly migrated checkpoint')
            # Explicit fresh start (video pilot only): every logical slot gets its
            # own local RNG stream, and the topology is still 3 ranks x 4 slots.
            executor.slot_rng = {slot: local_rng(device) for slot in SLOT_MAPPING[str(rank)]}
        executor.versions = parameter_versions(head)
        executor.head_sha = head_digest(head)
        executor.barrier(optimizer_step, start)
        if rank == 0:
            (out/'config.json').write_text(json.dumps(dict(vars(args), resume_identity=metadata,
                implementation='pure_block_three_card_four_slot_v1', block_native=True, diagnostic_only=args.smoke,
                training_identity=dict(metadata, head_version=dict(head_sha256=executor.head_sha,
                    optimizer_step=optimizer_step, schedule_cursor=start))), indent=2)+'\n')
        for cursor in range(start, args.updates):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            update_started = time.perf_counter()
            stop = bool(args.deadline_epoch and __import__('time').time() >= args.deadline_epoch - 900)
            if executor.announce(stop if rank == 0 else None):
                break
            head.zero_grad(set_to_none=True)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            traces, denominator, policy, records, wire = executor.collect(
                target, processor, rows, schedule[cursor], args, optimizer_step, cursor)
            numerator, norm, order = 0., None, []
            error = None
            if rank == 0:
                try:
                    for slot in range(len(schedule[cursor])):
                        trace = move_trace(traces[slot], device)
                        for anchor, stat in enumerate(runtime.score(target, head, trace, draft_fn=adapters.draft)):
                            if not stat.numerator.requires_grad or stat.numerator.grad_fn is None:
                                raise RuntimeError('student replay graph missing')
                            (stat.numerator / denominator).backward()
                            numerator += float(stat.numerator.detach())
                            order.append([slot, anchor, *traces[slot]['plan'][anchor]])
                        del trace
                    optimizer.sync_grads()
                    norm = optimizer.grad_norm()
                    if not math.isfinite(norm):
                        raise RuntimeError('gradient must be finite')
                    lr = (args.lr if args.lr_schedule == 'constant' else
                          args.lr * .5 * (1 + math.cos(math.pi * cursor / args.updates)))
                    optimizer.set_lr(lr)
                except Exception as exc:
                    error = repr(exc)
            executor.phase_check(error)
            if not args.no_update:
                if rank == 0:
                    budget_event(args.experiment_budget, args.experiment_case, 'reserve', cursor,
                        dict(entry=getattr(args, 'experiment_entry', 'scienceqa_pure_block'), seed=args.seed,
                             config_sha256=metadata['config_sha256']))
                    optimizer.step()
                    budget_event(args.experiment_budget, args.experiment_case, 'complete', cursor)
                optimizer_step += 1
            executor.commit(optimizer_step, cursor + int(not args.no_update), not args.no_update)
            if rank == 0:
                executor.slot_rng[0] = local_rng(device)
            all_rngs = {slot: state for bundle in executor.gather(executor.slot_rng)
                        for slot, state in bundle.items()}
            rngs = [all_rngs[slot] for slot in range(SLOTS)]
            torch.cuda.synchronize()
            telemetry = executor.gather(dict(rank=rank, seconds=time.perf_counter()-update_started,
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved()))
            error = None
            if rank == 0:
                try:
                    record = dict(update=cursor+1, optimizer_step=optimizer_step, optimizer_steps=optimizer_step,
                        tasks=[i['task'] for i in schedule[cursor]], loss=numerator/denominator,
                        denominator=denominator, denominator_hex=denominator.hex(),
                        slot_denominators=records, gradient_norm=norm, canonical_order=order,
                        trace_wire_payload_bytes=wire, head_sha256=executor.head_sha,
                        diagnostic_only=args.smoke, smoke=args.smoke, opd_mode=OPD_MODE,
                        teacher_verify_impl=args.teacher_verify_impl,
                        block_native=True, elapsed_seconds=max(t['seconds'] for t in telemetry),
                        peak_allocated=max(t['peak_allocated'] for t in telemetry),
                        per_rank_telemetry=telemetry,
                        target_forward_calls=sum(len(traces[s]['rounds']) + 2 for s in traces),
                        target_input_tokens=sum(traces[s]['prompt_len'] + 16*len(traces[s]['rounds']) + 1 for s in traces),
                        rounds=[len(traces[s]['rounds']) for s in range(len(schedule[cursor]))],
                        anchor_counts=[len(traces[s]['plan']) for s in range(len(schedule[cursor]))])
                    with (out/'log.jsonl').open('a') as stream:
                        stream.write(json.dumps(record)+'\n')
                    print(json.dumps(record), flush=True)
                except Exception as exc:
                    error = repr(exc)
            executor.phase_check(error)
            stop = bool((args.stop_file and Path(args.stop_file).exists()) or (args.stop_after and cursor+1-start >= args.stop_after)
                or (args.deadline_epoch and __import__('time').time() >= args.deadline_epoch-900))
            stop = executor.announce(stop if rank == 0 else None)
            if not args.no_update and ((cursor+1) % args.save_every == 0 or cursor+1 == args.updates or stop):
                if rank == 0:
                    training_resume.save_atomic_checkpoint(out/f'step{cursor+1}', head, dict(
                        head=head.state_dict(), master_optimizer=optimizer.state_dict(),
                        update=cursor+1, resume_schema=2, update_step=cursor+1,
                        data_cursor=cursor+1, resume_identity=metadata, rng=rngs[0],
                        diagnostic_only=args.smoke, smoke=args.smoke, opd_mode=OPD_MODE, block_native=True,
                        teacher_verify_impl=args.teacher_verify_impl,
                        training_identity=dict(metadata, head_version=dict(head_sha256=executor.head_sha,
                            optimizer_step=optimizer_step, schedule_cursor=cursor+1)),
                        parallel_state=dict(world_size=WORLD,
                            implementation='pure_block_three_card_four_slot_v1', slot_mapping=SLOT_MAPPING,
                            optimizer_step=optimizer_step, schedule_cursor=cursor+1,
                            head_sha256=executor.head_sha, per_slot_rng=rngs)))
                dist.barrier()
            if args.no_update or stop:
                break
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    from .cli import parser
    ap = parser()
    for action in ap._actions:
        if action.dest == 'parallel_rollout':
            action.choices = [3]
            action.default = 3
    ap.add_argument('--epoch-updates', type=int, default=1182)
    args = ap.parse_args()
    if args.smoke:
        ap.error('use real continuation updates; no separate smoke budget')
    if args.dry_run:
        print(json.dumps(dict(world_size=WORLD, slot_map=SLOT_MAPPING, qa_per_update=4, epoch_updates=args.epoch_updates)))
        return
    if not args.resume_from or not args.experiment_budget or not args.experiment_case:
        ap.error('explicit migrated resume checkpoint and continuation budget required')
    if args.epoch_updates != 1182 or args.updates != 3546:
        ap.error('this continuation is frozen to ScienceQA 3 epochs / 3546 total updates')
    run(args)

if __name__ == '__main__':
    main()
