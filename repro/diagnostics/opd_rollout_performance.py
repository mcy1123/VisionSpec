"""Bounded diagnostic: serial, ordered-anchor, or rollout-only parallelism.
The production trainer is not changed. All student graphs are canonical replays.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
import argparse
import datetime
import gc
import hashlib
import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from shared import common, training_resume
from shared.master_optim import MasterWeightAdamW
from opd_official_kl import adapters, runtime
from opd_official_kl.trainer import move_trace
from opd_official_kl.efficiency import (bind_trace, denominator_record,
    global_denominator, capture_gradients, compare_gradients)
from opd_official_kl.ordered_parallel import OrderedAnchorGradients, broadcast_head
from opd_official_kl.rollout_parallel import (tensors, raw_digest, tree_digest,
    head_digest, parameter_versions, check_versions, validate_envelope,
    send_trace, receive_trace, budget_event)
from opd_serial_candidate import compare_trace
from opd_efficiency_gate import BASE, write


class HashWriter:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()
        self.size = 0
    def write(self, data):
        self.hash.update(data)
        n = self.stream.write(data)
        self.size += n
        return n
    def flush(self):
        self.stream.flush()


def save_hashed(value, path):
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix('.pending')
    with tmp.open('wb') as stream:
        writer = HashWriter(stream)
        torch.save(value, writer)
        writer.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    write(path.with_suffix(path.suffix + '.sha256.json'), dict(
        sha256=writer.hash.hexdigest(), bytes=writer.size))


def local_rng(device):
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                python=random.getstate(), torch_cuda=[torch.cuda.get_rng_state(device)])


def restore_local_rng(state, device):
    training_resume.restore_rng(dict(state, torch_cuda=[]))
    if len(state['torch_cuda']) != 1:
        raise ValueError('exactly one local CUDA RNG state required')
    torch.cuda.set_rng_state(state['torch_cuda'][0], device)


class MeasuredAnchor(OrderedAnchorGradients):
    def __init__(self, *args):
        super().__init__(*args)
        self.sent_bytes = 0
        self.sent_anchors = 0
    def contribute(self, slot, index):
        if self.rank:
            self.sent_bytes += sum(p.grad.numel() * p.grad.element_size()
                                   for p in self.params.values() if p.grad is not None)
            self.sent_anchors += 1
        super().contribute(slot, index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--variant', choices=['serial', 'anchor', 'rollout'], required=True)
    ap.add_argument('--kernel', choices=['deterministic', 'default'], default='deterministic')
    ap.add_argument('--checkpoint-step', type=int, choices=[90, 120], default=120)
    ap.add_argument('--schedule-start', type=int, default=119)
    ap.add_argument('--updates', type=int, default=0)
    ap.add_argument('--budget', type=Path)
    ap.add_argument('--reference', type=Path)
    ap.add_argument('--legacy-gradient', type=Path)
    ap.add_argument('--legacy-traces', type=Path)
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--warmup', action='store_true')
    ap.add_argument('--save-state', action='store_true')
    args = ap.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if world != (1 if args.variant == 'serial' else 4):
        raise ValueError('one rank serial / four ranks parallel only')
    if args.updates < 0 or args.updates > 4 or (args.updates and not args.budget):
        raise ValueError('updates need a bounded authorization ledger')
    deterministic = args.kernel == 'deterministic'
    if deterministic != (os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'):
        raise ValueError('kernel/environment mismatch')
    training_resume.validate_output(args.out)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(deterministic)
    torch.cuda.set_device(local)
    device = f'cuda:{local}'
    distributed = world > 1
    if distributed:
        dist.init_process_group('gloo', timeout=datetime.timedelta(minutes=20))
    def barrier():
        if distributed:
            dist.barrier()
    def gather(value):
        if not distributed:
            return [value]
        result = [None] * world
        dist.all_gather_object(result, value)
        return result
    def announce(value):
        box = [value]
        if distributed:
            dist.broadcast_object_list(box, 0)
        return box[0]
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=False)
        write(args.out / 'RUN_STATUS.json', dict(status='LOADING', optimizer_steps=0))
    barrier()
    frozen = json.loads((args.run / 'INPUTS.json').read_text())
    cfg = frozen['config']
    import pandas as pd
    rows = pd.read_parquet(cfg['pool']).set_index('task', verify_integrity=True)
    schedule = [json.loads(x) for x in Path(cfg['schedule']).read_text().splitlines() if x.strip()]
    schedule_by_update = {}
    for item in schedule:
        schedule_by_update.setdefault(item['update'], []).append(item)
    for items in schedule_by_update.values():
        items.sort(key=lambda x: x['slot'])
    recipe = dict(config_sha256=frozen['config_sha256'],
                  schedule_sha256=frozen['hashes'][cfg['schedule']],
                  pool_sha256=frozen['hashes'][cfg['pool']],
                  source_hashes=dict(frozen['canonical_sources'], **frozen['candidate_sources']))
    if rank == 0:
        for path in [cfg['pool'], cfg['schedule'], *recipe['source_hashes']]:
            expected = frozen['hashes'].get(path, recipe['source_hashes'].get(path))
            if training_resume.file_hash(path) != expected:
                raise RuntimeError('frozen input/source changed: ' + path)
    target, processor = common.load_target(device=device, dtype=torch.float32)
    target.eval().requires_grad_(False)
    head = common.load_head(str(BASE / f'step{args.checkpoint_step}'),
                            device=device, dtype=torch.bfloat16).eval()
    head.requires_grad_(args.variant == 'anchor' or rank == 0)
    if len(list(head.named_parameters())) != 58:
        raise ValueError('expected original 58 parameter Head')
    checkpoint = args.resume or BASE / f'step{args.checkpoint_step}' / 'training_state.pt'
    initial = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=False)
    if args.resume:
        if (initial.get('diagnostic_schema') != 'rollout_only_v1'
            or not initial.get('diagnostic_only')
            or initial['variant'] != args.variant or initial['world_size'] != world
            or initial['recipe_identity'] != recipe
            or initial['schedule_cursor'] != args.schedule_start):
            raise ValueError('resume topology/cursor/recipe mismatch')
        rng = initial['per_rank_rng'][rank]
        optimizer_step = initial['optimizer_step']
    else:
        rng = initial['rng']
        optimizer_step = args.checkpoint_step
        if initial['update_step'] != args.checkpoint_step or initial['data_cursor'] != args.checkpoint_step:
            raise ValueError('original full checkpoint cursor mismatch')
    head.load_state_dict(initial['head'])
    optimizer = None
    if rank == 0 and args.updates:
        optimizer = MasterWeightAdamW(head, lr=cfg['lr'], betas=common.BETAS,
            weight_decay=common.WEIGHT_DECAY, where='cpu', clip=cfg['clip'])
        optimizer.load_state_dict(initial['master_optimizer'])
    nccl = None
    if distributed:
        nccl = dist.new_group(backend='nccl', timeout=datetime.timedelta(minutes=20))
        dist.broadcast(torch.zeros(1, device=device), 0, group=nccl)
    restore_local_rng(rng, device)
    current_hash = head_digest(head)
    check_versions(gather(dict(head_sha256=current_hash, optimizer_step=optimizer_step,
                               schedule_cursor=args.schedule_start)))
    expected_versions = parameter_versions(head)
    counts = {}
    def counter(name):
        def hook(module, inputs):
            counts[name] = counts.get(name, 0) + 1
        return hook
    target.model.register_forward_pre_hook(counter('target_model'))
    target.lm_head.register_forward_pre_hook(counter('lm_head'))
    head.register_forward_pre_hook(counter('draft_head'))
    selected = []
    original_pool = runtime.pool
    if deterministic:
        def audited_pool(q, p, rq, rp, offsets, **kw):
            selected.append(dict(response_q=q.detach().cpu(), response_p=p.detach().cpu(),
                rejected_q=rq.detach().cpu(), rejected_p=rp.detach().cpu(), offsets=offsets.detach().cpu()))
            return original_pool(q, p, rq, rp, offsets, **kw)
        runtime.pool = audited_pool

    def cycle(cursor, do_step, index, warmup=False):
        nonlocal current_hash, expected_versions, optimizer_step, counts
        items = schedule_by_update[cursor]
        if len(items) != 4 or [x['slot'] for x in items] != list(range(4)):
            raise ValueError('frozen four-slot schedule mismatch')
        owned = list(range(4)) if world == 1 else [rank]
        if parameter_versions(head) != expected_versions:
            raise RuntimeError('Head mutated outside the authorized version barrier')
        policy = dict(head_sha256=current_hash, optimizer_step=optimizer_step, schedule_cursor=cursor)
        selected.clear()
        head.zero_grad(set_to_none=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        reducer = MeasuredAnchor(head, nccl, rank, world) if args.variant == 'anchor' else None
        counts = {}
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); barrier()
        start = time.perf_counter()
        check_versions(gather(policy))
        traces = {}
        d2h = 0
        for slot in owned:
            row = rows.loc[items[slot]['task']].to_dict()
            if hashlib.sha256(row['prompt'].encode()).hexdigest() != items[slot]['prompt_sha256']:
                raise RuntimeError('frozen schedule/pool prompt mismatch')
            inputs = adapters.encode(processor, row, 'image', max_pixels=cfg['max_pixels'])
            tr = adapters.collect(target, head, inputs, max_new_tokens=cfg['max_new_tokens'],
                max_seq_len=cfg['max_seq_len'], teacher_verify_impl='serial_reference',
                batch_teacher_ids=False, preallocate_context=False)
            d2h += sum(t.numel() * t.element_size() for t in tensors(tr))
            tr = move_trace(tr, 'cpu')
            identity = dict(checkpoint_sha256=current_hash, task=items[slot]['task'], slot=slot,
                prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(),
                config_sha256=frozen['config_sha256'])
            bind_trace(tr, identity)
            tr['policy_identity'] = policy
            validate_envelope(tr, identity, policy)
            traces[slot] = tr
            del inputs, tr
        torch.cuda.synchronize(); barrier()
        rollout_end = time.perf_counter()
        wire = 0
        if args.variant == 'rollout':
            if rank:
                wire = send_trace(traces[rank])
            else:
                for source in range(1, 4):
                    tr, n = receive_trace(source)
                    row = rows.loc[items[source]['task']].to_dict()
                    ident = dict(checkpoint_sha256=current_hash, task=items[source]['task'], slot=source,
                        prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(),
                        config_sha256=frozen['config_sha256'])
                    validate_envelope(tr, ident, policy)
                    traces[source] = tr
        records = [denominator_record(traces[s], s, items[s]['task']) for s in owned]
        records = [x for group in gather(records) for x in group]
        denom = global_denominator(records, [x['task'] for x in items])
        if announce(denom.hex() if rank == 0 else None) != denom.hex():
            raise RuntimeError('global denominator binary64 differs')
        if parameter_versions(head) != expected_versions:
            raise RuntimeError('Head changed during rollout/transfer')
        barrier()
        transfer_end = time.perf_counter()
        order = []
        numerator = 0.
        h2d = 0
        def score_slot(slot):
            nonlocal numerator, h2d
            tr = traces[slot]
            h2d += sum(t.numel() * t.element_size() for t in tensors(tr))
            gpu = move_trace(tr, device)
            for anchor, stat in enumerate(runtime.score(target, head, gpu, draft_fn=adapters.draft)):
                if not stat.numerator.requires_grad or stat.numerator.grad_fn is None:
                    raise RuntimeError('student replay gradient graph missing')
                (stat.numerator / denom).backward()
                numerator += float(stat.numerator.detach())
                order.append([slot, anchor, *tr['plan'][anchor]])
                if reducer is not None:
                    reducer.contribute(slot, anchor)
            del gpu
        if reducer is not None:
            # Rank0 lacks remote traces in the anchor baseline: exchange plan lengths.
            lengths = [len(traces[s]['plan']) for s in owned]
            all_lengths = [x for group in gather(lengths) for x in group]
            for slot in range(4):
                if slot == rank:
                    score_slot(slot)
                elif rank == 0:
                    reducer.receive_slot(slot, all_lengths[slot])
        elif rank == 0:
            for slot in range(4):
                score_slot(slot)
        torch.cuda.synchronize(); barrier()
        backward_end = time.perf_counter()
        grad = reducer.finish() if reducer is not None and rank == 0 else None
        norm = None
        if rank == 0 and optimizer is not None:
            if grad is not None:
                optimizer.set_external_master_grads(grad)
            else:
                optimizer.sync_grads()
            norm = optimizer.grad_norm()
            if not math.isfinite(norm) or norm == 0:
                raise RuntimeError('invalid preclip gradient norm')
        barrier()
        sync_end = time.perf_counter()
        if rank == 0 and do_step:
            expected = dict(variant=args.variant, kernel=args.kernel,
                checkpoint_step=args.checkpoint_step, schedule_start=args.schedule_start,
                updates=args.updates)
            budget_event(args.budget, args.case, 'reserve', index, expected)
            optimizer.step()
            budget_event(args.budget, args.case, 'complete', index)
        barrier()
        optimizer_end = time.perf_counter()
        if distributed and (do_step or warmup):
            broadcast_head(head, nccl)
        if do_step:
            optimizer_step += 1
        current_hash = head_digest(head)
        expected_versions = parameter_versions(head)
        check_versions(gather(dict(head_sha256=current_hash, optimizer_step=optimizer_step,
                                   schedule_cursor=cursor + int(do_step))))
        torch.cuda.synchronize(); barrier()
        finish = time.perf_counter()
        timing = dict(rollout=rollout_end-start, trace_transfer_and_denominator=transfer_end-rollout_end,
            scoring_backward=backward_end-transfer_end, master_sync_and_norm=sync_end-backward_end,
            optimizer=optimizer_end-sync_end, broadcast_and_version_barrier=finish-optimizer_end,
            total=finish-start)
        reports = gather(dict(rank=rank, timing=timing, forward_counts=counts,
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            trace_d2h_tensor_bytes=d2h, trace_h2d_tensor_bytes=h2d,
            trace_wire_payload_bytes=wire,
            gradient_wire_payload_bytes=reducer.sent_bytes if reducer is not None else 0,
            gradient_remote_anchors=reducer.sent_anchors if reducer is not None else 0,
            loss_numerator=numerator, order=order))
        if warmup:
            return None
        folder = args.out / f'cursor{cursor:03d}'
        if rank == 0:
            folder.mkdir(exist_ok=False)
        barrier()
        # Trace audit and disk IO are deliberately outside every timing window.
        if args.variant == 'anchor':
            alltraces = gather(traces)
            if rank == 0:
                traces = {s:t for group in alltraces for s,t in group.items()}
        if rank == 0:
            for s in range(4):
                validate_envelope(traces[s], traces[s]['efficiency_identity'], policy)
            canonical_order = [[s, i, *p] for s in range(4) for i,p in enumerate(traces[s]['plan'])]
            observed_order = sorted([x for report in reports for x in report['order']], key=lambda x:(x[0],x[1]))
            if observed_order != canonical_order:
                raise RuntimeError('anchor coverage/order mismatch')
            if args.variant != 'anchor' and reports[0]['order'] != canonical_order:
                raise RuntimeError('rank0 canonical backward order changed')
            if grad is None:
                grad = capture_gradients(head)
            if len(grad) != 58 or any(g is None for g in grad.values()):
                raise RuntimeError('expected complete 58 parameter gradient graph')
            gradient_signature = tree_digest(grad)
            selected_all = [x for group in gather(selected) for x in group] if distributed else selected
        else:
            if distributed:
                gather(selected)
            gradient_signature = None
        rngs = gather(local_rng(device))
        if rank == 0:
            signature = dict(head=tree_digest(head.state_dict()), gradient=gradient_signature,
                master=tree_digest(optimizer.master) if optimizer else None,
                adam=tree_digest(optimizer._opt.state_dict()) if optimizer else None,
                rng=[tree_digest(x) for x in rngs])
            reference_pass = None
            if args.reference:
                ref_folder = args.reference / folder.name
                ref_traces = torch.load(ref_folder / 'traces.pt', map_location='cpu', weights_only=False, mmap=True)
                for s in range(4):
                    compare_trace(ref_traces[s], traces[s])
                    if ref_traces[s]['policy_identity'] != traces[s]['policy_identity']:
                        raise RuntimeError('policy identity differs from serial')
                ref_signature = json.loads((ref_folder/'SIGNATURE.json').read_text())
                keys = (['gradient'] if optimizer is None else ['gradient','head','master','adam']) if deterministic else []
                for key in keys:
                    if signature[key] != ref_signature[key]:
                        save_hashed(grad, folder/'FAILED_GRADIENTS.pt')
                        raise RuntimeError('deterministic exact gate failed: '+key)
                if any(x != ref_signature['rng'][0] for x in signature['rng']):
                    raise RuntimeError('per-rank common RNG differs from serial reference')
                ref_meta = json.loads((ref_folder/'RESULT.json').read_text())
                if ref_meta['denominator_hex'] != denom.hex() or ref_meta['canonical_order'] != canonical_order:
                    raise RuntimeError('denominator/anchor metadata mismatch')
                if deterministic and tree_digest(selected_all) != ref_meta['selected_logp_signatures']:
                    raise RuntimeError('selected student/teacher logp differs from serial')
                reference_pass = True
            if args.legacy_gradient:
                original = torch.load(args.legacy_gradient, map_location='cpu', weights_only=False, mmap=True)
                if tree_digest(original) != gradient_signature:
                    raise RuntimeError('new serial gate differs from prior deterministic legacy')
            if args.legacy_traces:
                original = torch.load(args.legacy_traces, map_location='cpu', weights_only=False)
                for s in range(4):
                    ref = dict(original[s], efficiency_identity=traces[s]['efficiency_identity'])
                    compare_trace(ref, traces[s])
            maximum = max(report['timing']['total'] for report in reports)
            result = dict(diagnostic_only=True, case=args.case, variant=args.variant, kernel=args.kernel,
                schedule_cursor=cursor, optimizer_step=optimizer_step, optimizer_steps=int(do_step),
                denominator_hex=denom.hex(), records=records, policy_before=policy, head_after=current_hash,
                reference_exact=reference_pass, canonical_order=canonical_order, gradient_tensors=58,
                reference_scope='trace/denominator/order/gradients/state/RNG' if deterministic else 'trace/denominator/order/RNG only',
                default_gradient_gate=False,
                preclip_norm=norm, loss=sum(x['loss_numerator'] for x in reports)/denom,
                wall_seconds=maximum, gpu_seconds=world*maximum, reports=reports,
                selected_logp_signatures=tree_digest(selected_all) if deterministic else None,
                head_broadcast_tensor_bytes=sum(p.numel()*p.element_size() for p in head.parameters()) if distributed and do_step else 0,
                head_broadcast_recipients=world-1, optimizer_steps_started=int(do_step))
            if not math.isfinite(result['loss']):
                raise RuntimeError('nonfinite full-update loss')
            if optimizer:
                result['adam_steps'] = sorted({float(x['step']) for x in optimizer._opt.state.values()})
                if result['adam_steps'] != [float(optimizer_step)]:
                    raise RuntimeError('optimizer step count differs from authorized update count')
            write(folder/'SIGNATURE.json',signature)
            write(folder/'RESULT.json', result)
            save_hashed([traces[s] for s in range(4)], folder/'traces.pt')
            if deterministic:
                save_hashed(grad, folder/'gradients.pt')
            print(json.dumps(dict(case=args.case, cursor=cursor, seconds=maximum,
                exact=reference_pass, optimizer_steps=int(do_step), loss=result['loss'])),flush=True)
            write(args.out/'RUN_STATUS.json', dict(status='RUNNING', optimizer_steps=index+int(do_step)))
        barrier()
        return rngs

    if args.warmup:
        cycle(args.schedule_start, False, -1, warmup=True)
        head.load_state_dict(initial['head'])
        if optimizer:
            optimizer.load_state_dict(initial['master_optimizer'])
        head.zero_grad(set_to_none=True)
        restore_local_rng(rng, device)
        optimizer_step = args.checkpoint_step
        current_hash = head_digest(head)
        expected_versions = parameter_versions(head)
        check_versions(gather(dict(head_sha256=current_hash, optimizer_step=optimizer_step,
                                   schedule_cursor=args.schedule_start)))
    # Retain original state only for warmup restoration; no mmap of future output.
    del initial
    gc.collect()
    final_rng = None
    for index in range(args.updates or 1):
        final_rng = cycle(args.schedule_start+index, bool(args.updates), index)
    if rank == 0 and args.updates and args.save_state:
        state_root = Path((args.run/'STATE_ROOT.txt').read_text().strip()) / args.case
        state_root.mkdir(parents=True,exist_ok=False)
        save_hashed(dict(diagnostic_schema='rollout_only_v1', diagnostic_only=True,
            variant=args.variant, world_size=world, recipe_identity=recipe,
            schedule_cursor=args.schedule_start+args.updates, optimizer_step=optimizer_step,
            head={n:p.detach().cpu() for n,p in head.state_dict().items()},
            master_optimizer=optimizer.state_dict(), per_rank_rng=final_rng,
            parent_checkpoint_step=args.checkpoint_step, case=args.case,
            slot_mapping={i:[i] for i in range(4)} if distributed else {0:list(range(4))}),
            state_root/'state.pt')
    if rank == 0:
        write(args.out/'RUN_STATUS.json', dict(status='COMPLETE', optimizer_steps=args.updates,
            diagnostic_only=True, next_schedule_cursor=args.schedule_start+args.updates))
    barrier()
    if distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
