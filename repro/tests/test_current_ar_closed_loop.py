
# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()

import unittest
import torch
from diagnostics.current_ar_closed_loop import leading, walk, candidates
from glance.tree import build_budget_tree


class CurrentARTests(unittest.TestCase):
    def test_leading_stops_on_first_mismatch(self):
        self.assertEqual(leading([True,True,False,True]),2)
        self.assertEqual(leading([]),0)

    def test_tree_accepts_non_top1_branch(self):
        nodes=build_budget_tree([[1,2],[3,4]],[[-.1,-.2],[-.1,-.2]],6,2)
        logits=torch.full((len(nodes)+1,8),-10.)
        logits[0,2]=10
        for i,n in enumerate(nodes):logits[i+1,4 if n.depth==1 else 7]=10
        path,bonus=walk(nodes,logits)
        self.assertEqual([nodes[i].token for i in path],[2,4])
        self.assertEqual(bonus,7)

    def test_actual_top8_stored_separately_from_top32(self):
        q=torch.randn(15,64);c=candidates(q);lp=q.log_softmax(-1)
        self.assertTrue(torch.equal(c['ids8'],lp.topk(8).indices))
        self.assertTrue(torch.equal(c['val8'],lp.topk(8).values))
        self.assertTrue(torch.equal(c['ids32'],lp.topk(32).indices))

    def test_no_candidate_matching_is_zero_draft_plus_bonus(self):
        nodes=build_budget_tree([[1]],[[-.1]],1,1)
        x=torch.zeros(2,8);x[:,5]=10
        self.assertEqual(walk(nodes,x),([],5))


if __name__=='__main__':unittest.main()
