"""Frozen-source numerical/gradient and real shared runtime CPU gates."""

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from typing import Optional
import torch
from opd_official_kl import loss as L, runtime as R, plan as P

ROOT = _REPRO_ROOT.parent
RUN = Path((ROOT / ".official_kl_audit_run").read_text().strip())
REF = RUN / "audit/official_reference"


def source(suffix):
    files = list(REF.rglob(suffix))
    if suffix == "losses.py":
        files = [p for p in files if p.parent.name == "distillation"]
    if len(files) != 1:
        raise RuntimeError((suffix, files))
    return files[0]


def extract(suffix, name):
    path = source(suffix)
    tree = ast.parse(path.read_text())
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    fn.decorator_list = []
    ns = dict(torch=torch, math=math, Optional=Optional, DistillationLossConfig=NS)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


class LossTests(unittest.TestCase):
    def test_official_forward_and_gradient(self):
        bern = extract("losses.py", "_local_bernoulli_forward_kl")
        k3 = extract("core_algos.py", "kl_penalty_forward")
        points = torch.tensor(
            [-float("inf"), -100, -80, -21, -20, -11, -10, -1, -1e-8, 0.0]
        )
        q = points.repeat_interleave(len(points)).requires_grad_()
        p = points.repeat(len(points)).clamp_min(-100).requires_grad_()
        # -inf - -inf is undefined; teacher is finite on the supported source.
        for fn, ref in [
            (
                L.bernoulli_forward,
                lambda q, p: bern(
                    student_log_probs=q,
                    teacher_log_probs=p,
                    loss_config=NS(log_prob_min_clamp=-10.0),
                ).clamp(-10, 10),
            ),
            (L.k3, lambda q, p: k3(q, p, "k3")),
        ]:
            actual = fn(q, p)
            expected = ref(q, p.detach())
            self.assertTrue(torch.equal(actual, expected))
            g = torch.autograd.grad(actual.sum(), q, retain_graph=True)[0]
            gr = torch.autograd.grad(expected.sum(), q, retain_graph=True)[0]
            self.assertTrue(torch.equal(g, gr))
            self.assertTrue(torch.isfinite(g).all())
            self.assertIsNone(
                torch.autograd.grad(
                    actual.sum(), p, allow_unused=True, retain_graph=True
                )[0]
            )
        q = torch.tensor([-0.4], requires_grad=True)
        p = torch.tensor([-1.2])
        b = L.bernoulli_forward(q, p)
        ce = -q
        self.assertNotEqual(b.item(), ce.item())
        self.assertFalse(
            torch.equal(torch.autograd.grad(b, q)[0], torch.autograd.grad(ce, q)[0])
        )

    def test_global_response_count_includes_uncovered_tokens(self):
        q = torch.tensor([-1.0], requires_grad=True)
        p = torch.tensor([-0.3])
        empty = q[:0]
        stat = L.pool(
            q, p, empty, empty, torch.tensor([], dtype=torch.long), response_count=3
        )
        self.assertEqual(stat.denominator.item(), 3)
        self.assertTrue(torch.equal(stat.loss(), L.bernoulli_forward(q, p).sum() / 3))
        # Extract the actual official pooled branch, retaining global-count logic.
        tree = ast.parse(source("losses.py").read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "distillation_loss"
        )
        branch = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "rejected_draft_losses is None"
        )
        body = []
        for n in branch.orelse:
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript):
                break
            body.append(n)
        ns = dict(
            torch=torch,
            loss_config=NS(
                response_stream_weight=1.0, rejected_draft_stream_weight=2.0
            ),
            config=NS(global_batch_info={"batch_num_tokens": 3, "dp_size": 1}),
            effective_response_mask=torch.ones(1),
            response_loss_weights=torch.ones(1),
            rejected_draft_mask=torch.zeros(0),
            rejected_loss_weights=torch.zeros(0),
            distillation_losses=L.bernoulli_forward(q, p),
            rejected_draft_losses=empty,
            response_position_decay_applied=False,
            rejected_position_decay_applied=True,
            dp_group=None,
            _global_sum=lambda x, g: x,
            _scalar_like=lambda x, t: torch.as_tensor(
                x, device=t.device, dtype=t.dtype
            ),
        )
        exec(
            compile(ast.Module(body=body, type_ignores=[]), "official_pool", "exec"), ns
        )
        self.assertTrue(torch.equal(ns["distillation_loss"], stat.loss()))

    def test_pool_shards_and_ddp(self):
        q = torch.tensor([-1.0, -0.5, -3.0], requires_grad=True)
        p = torch.tensor([-0.2, -0.8, -2.0])
        rq = torch.tensor([-2.0, -4.0], requires_grad=True)
        rp = torch.tensor([-1.0, -0.1])
        o = torch.tensor([1, 15])
        whole = L.pool(q, p, rq, rp, o)
        a = L.pool(q[:1], p[:1], rq[:1], rp[:1], o[:1])
        b = L.pool(q[1:], p[1:], rq[1:], rp[1:], o[1:])
        self.assertTrue(torch.allclose((a + b).loss(), whole.loss()))
        self.assertTrue(
            torch.allclose(
                (a.loss(whole.denominator, 2) + b.loss(whole.denominator, 2)) / 2,
                whole.loss(),
            )
        )
        bern = extract("losses.py", "_local_bernoulli_forward_kl")
        k3 = extract("core_algos.py", "kl_penalty_forward")
        response = bern(
            student_log_probs=q,
            teacher_log_probs=p,
            loss_config=NS(log_prob_min_clamp=-10.0),
        ).clamp(-10, 10)
        w = 0.9 ** (o.float() - 1)
        expected = (response.sum() + 2 * (k3(rq, rp, "k3") * w).sum()) / (
            len(q) + 2 * w.sum()
        )
        self.assertTrue(torch.equal(whole.loss(), expected))
        for actual_g, expected_g in zip(
            torch.autograd.grad(whole.loss(), (q, rq), retain_graph=True),
            torch.autograd.grad(expected, (q, rq)),
        ):
            self.assertTrue(torch.equal(actual_g, expected_g))
        for qn, rn in [(0, 0), (3, 0), (0, 2)]:
            v = L.pool(q[:qn], p[:qn], rq[:rn], rp[:rn], o[:rn]).loss()
            self.assertTrue(torch.isfinite(v))
            v.backward(retain_graph=True)


class PlanTests(unittest.TestCase):
    def test_official_anchor_plan(self):
        fn = extract("dflash_student.py", "_build_opd_anchor_plan")
        for length in [0, 1, 2, 5, 10]:
            for bounds in [[], [0], [1, 3], [4, 8], [-1, 2, 99]]:
                for rejected in [[], [(2, 1), (4, 3), (2, 2), (6, 0), (5, 4)]]:
                    prompt = 3
                    anchors, counts, _, mask, _, _ = fn(
                        None,
                        input_ids=torch.zeros(1, prompt + length + 3, dtype=torch.long),
                        attention_mask=torch.tensor(
                            [[1] * (prompt + length) + [0] * 3]
                        ),
                        prompt_lengths=torch.tensor([prompt]),
                        response_lengths=torch.tensor([length]),
                        reject_token_indices=torch.tensor([bounds], dtype=torch.long),
                        draft_block_size=4,
                        rejected_draft_anchor_indices=torch.tensor(
                            [[x - prompt for x, o in rejected]], dtype=torch.long
                        ),
                        rejected_draft_offsets=torch.tensor(
                            [[o for x, o in rejected]], dtype=torch.long
                        ),
                        rejected_draft_mask=torch.ones(
                            1, len(rejected), dtype=torch.bool
                        ),
                    )
                    expected = list(zip(anchors[mask].tolist(), counts[mask].tolist()))
                    self.assertEqual(
                        P.response_plan(prompt, length, bounds, 4, rejected), expected
                    )

    def test_verify_eos_horizon(self):
        cases = [
            ([1, 2, 3], [4, 5, 6, 7], 8, [4], [1, 2, 3]),
            ([1, 2, 3], [1, 5, 6, 7], 8, [1, 5], [2, 3]),
            ([1, 2, 3], [1, 2, 3, 7], 8, [1, 2, 3, 7], []),
            ([1, 9, 3], [1, 9, 3, 7], 8, [1, 9], []),
            ([1, 2, 3], [1, 9, 3, 7], 8, [1, 9], [2, 3]),
            ([1, 2, 3], [1, 2, 3, 7], 2, [1, 2], []),
            ([9, 2, 3], [4, 5, 6, 7], 1, [4], [1, 2, 3]),
        ]
        for prop, t, budget, committed, rejected in cases:
            d = P.verify(prop, t, {9}, budget)
            self.assertEqual(d["committed"], committed)
            self.assertEqual(d["rejected_offsets"], rejected)


class OnlineTests(unittest.TestCase):
    def test_shared_runtime_and_gradients(self):
        from tests.test_opd_verified_rollout import ToyTarget
        from transformers import DynamicCache
        from glance.vl import VisionPositions

        target = ToyTarget()
        target.training = False
        traces = []
        for proposal in [[0, 0, 0], [5, 3, 6], [1, 9, 2]]:
            head = torch.nn.Linear(1, 1)
            head.eval()
            head.block_size = 4

            # Differentiable selected-token probabilities, deterministic policy.
            def draft(head, target, fused, **kw):
                logits = (
                    torch.nn.functional.one_hot(torch.tensor(proposal), 12).float() * 2
                )
                return (
                    logits
                    + head.weight.reshape(1, 1) * torch.arange(12).reshape(1, 12) * 0.01
                )

            tr = R.rollout(
                target,
                head,
                torch.tensor([[2, 3]]),
                VisionPositions(),
                draft_fn=draft,
                feature_fn=lambda hs: hs[0],
                cache_factory=DynamicCache,
                eos={9},
                max_new_tokens=12,
                max_seq_len=40,
            )
            expected = [2, 3]
            for _ in range(12):
                expected.append(sum(expected) % 7 + 1)
            self.assertEqual(tr["ids"].tolist(), expected)
            self.assertEqual(tr["final_cache_length"], len(expected))
            for r in tr["rounds"]:
                prefix = expected[: r["pos"] + 1]
                greedy = []
                logp = []
                for token in r["prop"]:
                    g = sum(prefix) % 7 + 1
                    greedy.append(g)
                    lp = (
                        torch.nn.functional.one_hot(torch.tensor(g), 12)
                        .float()
                        .log_softmax(-1)
                    )
                    logp.append(lp[token])
                    prefix.append(token)
                self.assertEqual(r["teacher_ids"][:-1], greedy)
                self.assertTrue(torch.equal(r["teacher_logp"], torch.stack(logp)))
            stats = list(R.score(target, head, tr, draft_fn=draft))
            total = sum(stats[1:], stats[0])
            self.assertAlmostEqual(
                total.denominator.item(), R.denominator(tr), places=4
            )
            total.loss().backward()
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(head.weight.grad.abs().sum(), 0)
            traces.append([r["pos"] for r in tr["rounds"]])
        self.assertNotEqual(
            traces[0], traces[1]
        )  # policy changes actually visited anchors
        self.assertNotEqual(
            traces[1], list(range(2, 13))
        )  # fixed AR dense anchors differ

    def test_both_adapters_execute_shared_rollout_score(self):
        from opd_official_kl import adapters
        from tests.test_opd_verified_rollout import ToyTarget
        from unittest.mock import patch

        target = ToyTarget()
        target.training = False
        head = torch.nn.Linear(1, 1)
        head.eval()
        head.block_size = 4

        def draft(head, target, fused, **kw):
            return (
                torch.nn.functional.one_hot(torch.tensor([5, 3, 6]), 12).float() * 2
                + head.weight.reshape(1, 1) * torch.arange(12) * 0.01
            )

        for pixels, grid in [
            ("pixel_values", "image_grid_thw"),
            ("pixel_values_videos", "video_grid_thw"),
        ]:
            inputs = {
                "input_ids": torch.tensor([[2, 3]]),
                pixels: torch.zeros(1, 3),
                grid: torch.tensor([[1, 1, 1]]),
            }
            with patch.object(adapters, "draft", draft):
                t = adapters.collect(
                    target, head, inputs, max_new_tokens=8, max_seq_len=40
                )
                stats = list(R.score(target, head, t, draft_fn=adapters.draft))
                loss = sum(x.numerator for x in stats) / R.denominator(t)
                head.zero_grad()
                loss.backward()
                self.assertTrue(torch.isfinite(head.weight.grad).all())
                # Padding is a fail-fast input error, never a fake visited state.
                with self.assertRaises(ValueError):
                    adapters.collect(
                        target,
                        head,
                        dict(inputs, attention_mask=torch.tensor([[1, 0]])),
                        max_new_tokens=8,
                        max_seq_len=40,
                    )
                # Replaying another policy must fail before a loss is accepted.
                next(r for r in t["rounds"] if r["pos"] in {p for p, n in t["plan"]})[
                    "prop"
                ][0] = 11
                with self.assertRaises(RuntimeError):
                    list(R.score(target, head, t, draft_fn=adapters.draft))

    def test_actual_dflash_geometry_cpu(self):
        from transformers import Qwen3Config
        from dflash.model import DFlashDraftModel
        from tests.test_opd_verified_rollout import ToyTarget
        from opd_official_kl import adapters

        cfg = Qwen3Config(
            hidden_size=12,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=12,
            max_position_embeddings=64,
            num_target_layers=36,
            block_size=4,
            dflash_config={"mask_token_id": 11, "target_layer_ids": [1, 9, 17, 25, 33]},
        )
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(9)
        head = DFlashDraftModel(cfg).to(torch.bfloat16).eval()
        target = ToyTarget()
        target.training = False
        target.lm_head = torch.nn.Linear(12, 12, bias=False)
        target.lm_head.weight.data.copy_(torch.eye(12))
        target.lm_head.requires_grad_(False)
        embed = torch.nn.Embedding(12, 12)
        embed.weight.data.copy_(torch.eye(12))
        embed.requires_grad_(False)
        target.get_input_embeddings = lambda: embed
        inputs = {
            "input_ids": torch.tensor([[2, 3]]),
            "pixel_values": torch.zeros(1, 3),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }
        t = adapters.collect(target, head, inputs, max_new_tokens=6, max_seq_len=40)
        stats = list(R.score(target, head, t, draft_fn=adapters.draft))
        loss = sum(x.numerator for x in stats) / R.denominator(t)
        loss.backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.float().abs().sum()) for g in grads), 0)
        self.assertTrue(all(p.grad is None for p in target.lm_head.parameters()))

    def test_adapter_dispatch(self):
        from opd_official_kl import adapters
        from unittest.mock import patch

        for modality, mod, name in [
            ("image", "scienceqa_image.imageencode", "encode_image_inputs"),
            ("video", "qa_opd_fair.videoencode", "encode_inputs"),
        ]:
            with patch(mod + "." + name, return_value={"encoded": modality}) as fn:
                self.assertEqual(
                    adapters.encode(None, {}, modality), {"encoded": modality}
                )
                fn.assert_called_once()
        # Both encoded modalities share the very same collect -> rollout -> score.
        self.assertIn("rollout", adapters.collect.__code__.co_names)
        head = torch.nn.Linear(1, 1)
        with patch(
            "shared.draft_geometry.draft_block_logits_mixed", return_value="ok"
        ) as fn:
            self.assertEqual(
                adapters.draft(
                    head, None, torch.zeros(1, 10, 2), anchor_pos=3, root_token=1
                ),
                "ok",
            )
            self.assertEqual(fn.call_args.args[2].shape, (1, 3, 2))


if __name__ == "__main__":
    unittest.main()
