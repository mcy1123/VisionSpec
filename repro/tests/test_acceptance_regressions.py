"""Acceptance regressions: distinguish objectives and preserve interrupted SFT state."""
import ast
import unittest
from pathlib import Path
import torch
from opd_official_kl import loss
from historical.opd_fullkl import full_kl_reverse

ROOT = Path(__file__).resolve().parents[2]


class AcceptanceTests(unittest.TestCase):
    def test_controlled_legacy_adapter_differs(self):
        # Same input, model and teacher; change only the controlled loss adapter.
        head = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[.2, -.3, .1], [.4, .5, -.2], [-.1, .3, .6]]))
        features = torch.tensor([[1., 2., -1.], [-2., .5, 1.]])
        teacher = torch.tensor([[.7, -.2, .1], [-.4, .3, .8]], requires_grad=True)
        logits = head(features)
        q, p = logits.log_softmax(-1), teacher.detach().log_softmax(-1)
        tokens = torch.tensor([0, 1])
        canonical = loss.pool(q[0, :1], p[0, :1], q[1, 1:2], p[1, 1:2], torch.tensor([1])).loss()
        legacy = (-q[0, 0] + 2 * full_kl_reverse(logits[1:], teacher.detach()[1:]).sum()) / 3
        full_forward = (p[0].exp() * (p[0] - q[0])).sum()
        bernoulli = loss.bernoulli_forward(q[0, :1], p[0, :1]).sum()
        gc = torch.autograd.grad(canonical, head.weight, retain_graph=True)[0]
        gl = torch.autograd.grad(legacy, head.weight, retain_graph=True)[0]
        gb = torch.autograd.grad(bernoulli, head.weight, retain_graph=True)[0]
        gf = torch.autograd.grad(full_forward, head.weight)[0]
        self.assertFalse(torch.equal(gc, gl))
        self.assertFalse(torch.equal(gb, gf))
        self.assertNotEqual(canonical.item(), legacy.item())
        self.assertIsNone(teacher.grad)

    def test_four_microbatches_with_unequal_counts(self):
        q = torch.linspace(-3, -.1, 12, requires_grad=True)
        p = torch.linspace(-.2, -2, 12)
        parts = []
        for start, end in [(0, 1), (1, 3), (3, 7), (7, 12)]:
            parts.append(loss.pool(q[start:end], p[start:end], q[start:end:2], p[start:end:2], torch.arange(1, end-start+1, 2)))
        whole = sum(parts[1:], parts[0])
        aggregate = sum(s.numerator / whole.denominator for s in parts)
        self.assertTrue(torch.allclose(aggregate, whole.loss(), atol=1e-7))
        g1 = torch.autograd.grad(aggregate, q, retain_graph=True)[0]
        g2 = torch.autograd.grad(whole.loss(), q)[0]
        self.assertTrue(torch.allclose(g1, g2, atol=1e-7))
        self.assertFalse(torch.allclose(sum(s.loss() for s in parts)/4, whole.loss()))

    def test_sft_checkpoint_saves_only_actual_update(self):
        for rel in ['scienceqa_image/train_sqa_sft.py', 'qa_opd_fair/train_qa_sft.py']:
            source = (ROOT/'repro'/rel).read_text()
            tree = ast.parse(source)
            calls = [ast.unparse(n) for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(
                         ('save_pretrained', 'save_atomic_checkpoint'))]
            self.assertEqual(len(calls), 2)  # Initial and full-state update checkpoint.
            self.assertNotIn('f"step{a.updates}"', source)
            self.assertIn('checkpoint already exists; refusing overwrite', source)
            self.assertIn('--resume-update requires --resume-from', source)


if __name__ == '__main__':
    unittest.main()
