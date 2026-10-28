"""ScienceQA image OPD: pure block-native + four GPUs, approximate method.

Legacy serial and 1+keep implementations are archived, not active alternatives.
Deployment decoder/verifier are unchanged. No equivalence claim is made.
"""
import argparse
import json
import os
from pathlib import Path
import sys

STATUS = dict(
    status="block_native_approximate",
    equivalence_level="not_claimed",
    objective="bernoulli_forward_kl_rejected_k3_block_native",
    anchor_mode="on_policy",
    source_HEAD="d6f0ec859321959785c65b59eb311fd25f042158",
    script_commit="d1476f5d00a5fb187eb1637dc3f0f610ddbe1e1d",
    forward_kl_weight=1.0,
    reverse_kl_weight=0.0,
    rejected_draft_use_reverse_kl=True,
    rejected_decay=0.9,
    response_weight=1.0,
    rejected_weight=2.0,
    log_prob_min_clamp=-10.0,
    loss_max_clamp=10.0,
    response_stride=1,
    random_response_anchor=False,
    first_token_only=False,
    response_position_decay=False,
    implementation="pure block-native teacher; four-rank rollout; rank0 ordered scoring",
)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true', help='print defaults without loading models or using a GPU')
    ap.add_argument('--modality', choices=['image','video'], default='image',
                    help='image = ScienceQA default; video is explicit only (see video_caption/)')
    for name in ['pool','schedule','init-from','out']:
        ap.add_argument('--'+name)
    ap.add_argument('--opd-mode', choices=['pure_block'], default='pure_block')
    ap.add_argument('--teacher-verify-impl', choices=['pure_block_native'], default='pure_block_native')
    ap.add_argument('--parallel-rollout', type=int, choices=[4], default=4,
                    help='default4; requires torchrun --nproc_per_node=4 and four visible GPUs; no fallback')
    ap.add_argument('--updates', type=int, default=1)
    ap.add_argument('--qa-per-update', type=int, choices=[4], default=4)
    ap.add_argument('--allow-partial-final', action='store_true', help='idle unused ranks only on final update; no repeated QA')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu', type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--lr-schedule', choices=['cosine','constant'], default='cosine')
    ap.add_argument('--clip', type=float, default=1.)
    ap.add_argument('--target-dtype', choices=['float32'], default='float32')
    ap.add_argument('--master-weights', choices=['cpu'], default='cpu')
    ap.add_argument('--time-mode', choices=['real','legacy'], default='real')
    ap.add_argument('--max-pixels', type=int, default=768)
    ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--max-seq-len', type=int, default=4096)
    ap.add_argument('--save-every', type=int, default=30)
    ap.add_argument('--resume-from', help='same-mode four-rank checkpoint only; smoke checkpoints cannot seed training')
    ap.add_argument('--resume-update', type=int)
    ap.add_argument('--stop-file', help='save and stop at the next completed update boundary')
    ap.add_argument('--stop-after', type=int, default=0)
    ap.add_argument('--deadline-epoch', type=float, default=0)
    ap.add_argument('--no-update', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='exactly one diagnostic update; cannot resume or become formal step121')
    ap.add_argument('--update-budget','--experiment-budget',dest='experiment_budget',
                    help='explicit bounded optimizer-attempt ledger')
    ap.add_argument('--update-case','--experiment-case',dest='experiment_case')
    return ap


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    if a.smoke:
        if a.resume_from or a.no_update or a.stop_after not in (0,1):
            ap.error('--smoke requires one new real update, without resume')
        a.stop_after = 1
    print(json.dumps(dict(STATUS,modality=a.modality,opd_mode=a.opd_mode,
        teacher_verify_impl=a.teacher_verify_impl,parallel_rollout=a.parallel_rollout,
        world_size=4,slot_map={str(i):[i] for i in range(4)},
        head_version='bound to Head SHA/optimizer step/cursor at each update',
        smoke=a.smoke,diagnostic_only=a.smoke,
        optimizer_steps=0 if a.dry_run or a.no_update else 'requested'),indent=2))
    if a.dry_run:
        return
    if min(a.updates,a.max_new_tokens,a.save_every)<=0:
        ap.error('positive update/token/checkpoint budgets required')
    if int(os.environ.get('WORLD_SIZE','1')) != 4:
        ap.error('four GPUs required: launch with torchrun --nproc_per_node=4; no automatic fallback')
    import torch
    if torch.cuda.device_count()<4:
        ap.error('four visible GPUs required for pure_block; no automatic fallback')
    for name in ['pool','schedule','init_from','out']:
        if not getattr(a,name):ap.error('--'+name.replace('_','-')+' required')
    if not a.no_update and (not a.experiment_budget or not a.experiment_case):
        ap.error('real updates require --update-budget and --update-case')
    repro=Path(__file__).resolve().parents[1];root=repro.parent
    sys.path[:0]=[str(repro),str(root/'GLANCE_repo'),str(root/'scripts/dflash_pkg/dflash-0.1.0')]
    from .trainer import run
    run(a)

if __name__=='__main__':
    main()
