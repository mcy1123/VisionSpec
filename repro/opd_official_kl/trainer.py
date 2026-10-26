"""ScienceQA pure-block training identity and four-rank dispatcher."""
import hashlib
import json
from pathlib import Path
import torch
from shared import training_resume
from .cli import STATUS

WORLD = 4
OPD_MODE = 'pure_block'
SLOT_MAPPING = {str(i): [i] for i in range(WORLD)}


def move_trace(trace, device):
    moved = {}
    for name, value in trace.items():
        if torch.is_tensor(value):
            moved[name] = value.to(device)
        elif name == 'rounds':
            moved[name] = [move_trace(row, device) for row in value]
        else:
            moved[name] = value
    return moved


def identity(args):
    ignored = {'out','gpu','resume_from','resume_update','stop_after','dry_run',
               'no_update','stop_file','save_every','deadline_epoch','experiment_budget','experiment_case'}
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


def run(args):
    if args.opd_mode != OPD_MODE or args.parallel_rollout != WORLD:
        raise ValueError('ScienceQA OPD requires pure_block and four GPUs; no fallback')
    from .parallel_trainer import run as run_parallel
    return run_parallel(args)
