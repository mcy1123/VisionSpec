#!/usr/bin/env python3
"""Explicit three-card fresh-start image OPD (clue/visual-corruption ablation)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opd_official_kl.cli import parser
from opd_official_kl import three_card_trainer

if __name__ == '__main__':
    ap = parser()
    for action in ap._actions:
        if action.dest == 'parallel_rollout':
            action.choices = [3]
            action.default = 3
    args = ap.parse_args()
    if not hasattr(args, 'epoch_updates'):
        args.epoch_updates = args.updates
    if not hasattr(args, 'allow_partial_final'):
        args.allow_partial_final = False
    three_card_trainer.run(args, allow_fresh_start=True)
