"""CPU contracts for the strict online state machine, without model downloads."""

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from shared import common
from historical import opd_train_video_fullkl as T
from historical import opd_verified_rollout as V


class ToyTarget:
    """Prefix-dependent oracle: retaining a rejected KV row changes the answer."""
    config = NS(eos_token_id=9)
    generation_config = NS(eos_token_id=[9, 10])

    def __init__(self):
        self.model = self
        self.lm_head = torch.nn.Identity()

    def get_rope_index(self, ids, *args):
        return torch.arange(ids.shape[1])[None, None].expand(3, 1, -1), torch.zeros(1, 1)

    def __call__(self, ids, *, past_key_values, **kwargs):
        cache = past_key_values
        rows = []
        for token in ids[0]:
            v = token.float().reshape(1, 1, 1, 1)
            cache.update(v, v, 0)
            prediction = (int(cache.layers[0].keys.sum()) % 7) + 1
            rows.append(torch.nn.functional.one_hot(torch.tensor(prediction), 12).float())
        h = torch.stack(rows)[None]
        return NS(past_key_values=cache, last_hidden_state=h,
                  hidden_states=tuple(h for _ in range(35)))


def draft(head, target, fused, **kw):
    return torch.nn.functional.one_hot(torch.tensor(head.proposal), 12).float()


class OnlineTests(unittest.TestCase):
    def test_decisions(self):
        cases = [
            ([1, 2, 3], [4, 5, 6, 7], 8, (0, [4])),
            ([1, 2, 3], [1, 5, 6, 7], 8, (1, [1, 5])),
            ([1, 2, 3], [1, 2, 3, 7], 8, (3, [1, 2, 3, 7])),
            ([1, 9, 3], [1, 9, 3, 7], 8, (2, [1, 9])),
            ([1, 2, 3], [1, 9, 3, 7], 8, (1, [1, 9])),
            ([1, 2, 3], [1, 2, 3, 9], 8, (3, [1, 2, 3, 9])),
            ([1, 2, 3], [1, 2, 3, 7], 2, (2, [1, 2])),
            ([1, 2, 3], [1, 2, 3, 7], 0, (0, [])),
        ]
        for prop, teacher, remaining, expected in cases:
            with self.subTest(prop=prop, teacher=teacher, remaining=remaining):
                n, tokens, valid = V.decision(prop, teacher, eos={9}, remaining=remaining)
                self.assertEqual((n, tokens), expected)
                self.assertEqual(sum(valid), min(remaining, prop.index(9) + 1 if 9 in prop else len(prop)))

    def test_state_matches_independent_ar(self):
        for prop in ([0, 0, 0], [1, 2, 3], [7, 7, 7], [1, 9, 2]):
            with self.subTest(prop=prop):
                target = ToyTarget()
                state, anchors = V.rollout(target, NS(block_size=4, proposal=prop),
                    torch.tensor([[2, 3]]), pixel_values_videos=None, video_grid_thw=None,
                    max_seq_len=40, max_new_tokens=12, max_rounds=0,
                    head_dtype=torch.float32, draft_fn=draft)
                expected = [2, 3]
                for _ in range(12):
                    expected.append(sum(expected) % 7 + 1)
                self.assertEqual(state['full_ids'].tolist(), expected)
                self.assertEqual(state['prefix_kv'].layers[0].keys.flatten().tolist(), expected)
                self.assertEqual(state['fused'].shape[1], len(expected))
                self.assertTrue(anchors)

    def test_round_limit(self):
        state, anchors = V.rollout(ToyTarget(), NS(block_size=4, proposal=[0]*3),
            torch.tensor([[2, 3]]), pixel_values_videos=None, video_grid_thw=None,
            max_seq_len=40, max_new_tokens=12, max_rounds=2,
            head_dtype=torch.float32, draft_fn=draft)
        self.assertEqual(len(anchors), 2)
        self.assertEqual(state['stop_reason'], 'round_limit')
        self.assertEqual(state['full_ids'].numel(), 5)

    def test_initial_eos_and_alternate_eos(self):
        target = ToyTarget()
        target.generation_config = NS(eos_token_id=[6, 10])
        state, anchors = V.rollout(target, NS(block_size=4, proposal=[0]*3),
            torch.tensor([[2, 3]]), pixel_values_videos=None, video_grid_thw=None,
            max_seq_len=40, max_new_tokens=12, max_rounds=0,
            head_dtype=torch.float32, draft_fn=draft)
        self.assertEqual(state['full_ids'].tolist(), [2, 3, 6])
        self.assertEqual(anchors, [])
        self.assertEqual(state['stop_reason'], 'eos')

    def test_short_horizon(self):
        state, anchors = V.rollout(ToyTarget(), NS(block_size=4, proposal=[0]*3),
            torch.tensor([[2, 3]]), pixel_values_videos=None, video_grid_thw=None,
            max_seq_len=4, max_new_tokens=12, max_rounds=0,
            head_dtype=torch.float32, draft_fn=draft)
        self.assertEqual(anchors, [])
        self.assertEqual(state['prefix_kv'].get_seq_length(), 3)

    def test_eos_commit_paths(self):
        for prop in ([5, 3, 0], [5, 4, 0], [5]):
            with self.subTest(prop=prop):
                target = ToyTarget()
                target.generation_config = NS(eos_token_id=[3, 10])
                state, anchors = V.rollout(target, NS(block_size=len(prop)+1, proposal=prop),
                    torch.tensor([[2, 3]]), pixel_values_videos=None, video_grid_thw=None,
                    max_seq_len=40, max_new_tokens=12, max_rounds=0,
                    head_dtype=torch.float32, draft_fn=draft)
                self.assertEqual(state['full_ids'].tolist(), [2, 3, 6, 5, 3])
                self.assertEqual(state['stop_reason'], 'eos')
                self.assertEqual(len(anchors), 1)
                self.assertEqual(state['prefix_kv'].layers[0].keys.flatten().tolist(), [2, 3, 6, 5, 3])

    def test_online_plan_does_not_require_old_future(self):
        a = dict(pos=3, prop=[1]*15, n_acc=0, valid=[True]*15,
                 teacher_ids=[2]*15, source='current_target')
        kept, _ = T.select_anchors([a], budget=1, max_seq_len=40,
                                   prompt_len=3, valid_end=4)
        self.assertEqual(kept, [a])
        plan = T.build_anchor_plan('toy', a, total=5, valid_end=4, prompt_len=3)
        self.assertEqual(len(plan.offsets_rej), 15)
        p = torch.nn.functional.one_hot(torch.tensor([2]*15), 3).float()
        q = torch.nn.functional.one_hot(torch.tensor([1]*15), 3).float()
        V.assert_replay(plan, p, q)
        for kind in ('label', 'teacher', 'student'):
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                if kind == 'label':
                    plan.n_acc = 1
                    V.assert_replay(plan, p, q)
                elif kind == 'teacher':
                    plan.n_acc = 0
                    V.assert_replay(plan, q, q)
                else:
                    V.assert_replay(plan, p, p)

    def test_prompt_ids_and_continuation_independence(self):
        class Processor:
            def apply_chat_template(self, *args, **kw):
                return 'prompt'
            def __call__(self, **kw):
                return dict(input_ids=torch.tensor([[2, 3]]),
                            pixel_values_videos=torch.empty(0), video_grid_thw=torch.empty(0))
        row = dict(question='q', cache_path='unused', fps=1)
        outputs = []
        with patch.object(T, 'draft_block_logits', draft):
            for suffix in ([1], [9, 9, 9], [999]*100):
                rec = dict(task='toy', prompt_len=2, input_ids=torch.tensor([2, 3]+suffix))
                trace, anchors = T.build_video_trace(ToyTarget(), NS(block_size=4, proposal=[0]*3),
                    Processor(), rec, row, device='cpu', max_seq_len=40, max_rounds=2)
                outputs.append((trace.full_ids.tolist(), anchors))
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[1], outputs[2])
            rec['input_ids'][0] = 5
            with self.assertRaisesRegex(ValueError, 'prompt token IDs'):
                T.build_video_trace(ToyTarget(), NS(block_size=4), Processor(), rec, row,
                                    device='cpu', max_seq_len=40, max_rounds=2)

    def test_default_is_online(self):
        a = T.parse_args(['--records','r','--pool','p','--init-from','h','--out','o'])
        self.assertEqual(a.rollout_mode, 'online')

    def test_missing_teacher_evidence_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'lacks current-target'):
            V.assert_replay(NS(teacher_ids=None), None, None)

    def test_dropout_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, 'eval mode'):
            V.rollout(ToyTarget(), NS(training=True), torch.tensor([[2, 3]]),
                pixel_values_videos=None, video_grid_thw=None, max_seq_len=40,
                max_new_tokens=12, max_rounds=0, head_dtype=None, draft_fn=draft)


if __name__ == '__main__':
    unittest.main(verbosity=2)
