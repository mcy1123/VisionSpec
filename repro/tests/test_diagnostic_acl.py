"""CPU tests for reference separation and terminal-safe local diagnostics."""

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()

import unittest
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(_REPRO_ROOT))
from shared import common
from glance.tree import build_budget_tree
from diagnostics.diagnostic_acl_metrics import part_of, paired, native_objective, distribution_metrics, condition_key, local_oracles


class DiagnosticTests(unittest.TestCase):
    def test_three_parts_and_mask(self):
        self.assertEqual([part_of(i,2) for i in range(1,5)],
                         ['accepted','accepted','first_rejected','strict_suffix'])
        self.assertIsNone(part_of(4,2,False))
        with self.assertRaises(ValueError):part_of(0,2)

    def test_normalisation_not_mean_of_anchors(self):
        aa=[dict(valid=[True,True],n_acc_anchor=1,fkl=[2.,0.],rkl=[0.,10.]),
            dict(valid=[True,True],n_acc_anchor=0,fkl=[0.,0.],rkl=[4.,6.])]
        got=native_objective(aa)
        self.assertAlmostEqual(got['loss'],(2+(8+4+4.8)/2.6)/2,6)

    def test_empty_streams(self):
        self.assertEqual(native_objective([])['loss'],0)
        aa=[dict(valid=[True],n_acc_anchor=1,fkl=[2.],rkl=[100.])]
        self.assertEqual(native_objective(aa)['loss'],2.)
        aa[0]['n_acc_anchor']=0
        self.assertEqual(native_objective(aa)['loss'],100.)

    def test_pairing_is_by_video_and_repeatable(self):
        a={'a':1.,'b':2.,'onlya':9.};b={'a':2.,'b':1.}
        x=paired(a,b)
        self.assertEqual(x,paired(a,b));self.assertEqual(x['n'],2)
        self.assertEqual(x['positive'],1);self.assertEqual(x['changed'],2)
        self.assertIsNone(paired({},b)['mean'])

    def test_rank_topk_logprob_and_repeatability(self):
        q=torch.arange(40,dtype=torch.float32)[None]
        p=torch.full_like(q,-5);p[0,35]=5
        a=distribution_metrics(p,q);b=distribution_metrics(p,q)
        self.assertEqual(a['rank'].item(),5)
        self.assertTrue(a['top8'].item());self.assertFalse(a['top1'].item())
        for k in a:self.assertTrue(torch.equal(a[k],b[k]),k)
        same=distribution_metrics(p,p)
        self.assertAlmostEqual(same['js'].item(),0.,places=7)
        self.assertAlmostEqual(same['fkl'].item(),0.)

    def test_tie_rank_is_not_topk_membership(self):
        q=torch.zeros(1,40)
        selected=q.topk(8).indices[0].tolist()
        token=next(i for i in range(40) if i not in selected)
        p=q.clone();p[0,token]=10
        m=distribution_metrics(p,q)
        self.assertEqual(m['rank'].item(),1)
        self.assertFalse(m['top8'].item())

    def test_cache_isolates_content_order_time_processor_intervention(self):
        d=dict(content_sha256='aaa',frame_indices=[0,1],timestamps=[0.,1.],
               processor={'fps':1},intervention='original')
        base=condition_key(**d)
        self.assertEqual(base,condition_key(**d))
        for key,value in [('content_sha256','bbb'),('frame_indices',[1,0]),
                          ('timestamps',[0.,2.]),('processor',{'fps':2}),
                          ('intervention','swap')]:
            self.assertNotEqual(base,condition_key(**{**d,key:value}))

    def oracle(self,ids,nodes,ref,actual,**kwargs):
        defaults=dict(budget=63,remaining_after_root=15,root_token=100,eos_id=99,
                      reference_verified=True)
        return local_oracles(ids,nodes,ref,actual,**{**defaults,**kwargs})

    def test_b1_is_one_candidate_not_block_chain(self):
        ids=[[i] for i in range(15)];lp=[[-.1]]*15
        self.assertEqual(len(build_budget_tree(ids,lp,1,1)),1)
        nodes=build_budget_tree(ids,lp,15,1)
        self.assertEqual(len(nodes),15)
        self.assertEqual([n.depth for n in nodes],list(range(1,16)))
        self.assertEqual([n.parent for n in nodes],list(range(-1,14)))

    def test_prediction_missing(self):
        x=self.oracle([[1],[2]],[],[3,2],[])
        self.assertEqual(x['reason'],'raw_candidate_missing')
        self.assertEqual(x['L_raw'],0)

    def test_selection_loss_and_prefix(self):
        nodes=[dict(parent=-1,depth=1,token=4)]
        x=self.oracle([[1,4],[2,5]],nodes,[1,2],[])
        self.assertEqual(x['L_raw'],2);self.assertEqual(x['L_candidate_oracle'],2)
        self.assertEqual(x['L_tree_oracle'],0);self.assertEqual(x['reason'],'tree_selection_loss')

    def test_budget_excludes_root(self):
        nodes=[dict(parent=-1,depth=1,token=1)]
        x=self.oracle([[1],[2]],nodes,[1,2],[1],budget=1)
        self.assertEqual(x['verify_nodes'],2)
        self.assertEqual(x['L_candidate_oracle'],1)
        self.assertEqual(x['reason'],'budget_constraint')

    def test_bonus_exclusion_and_length_clipping(self):
        nodes=[dict(parent=-1,depth=1,token=1),dict(parent=0,depth=2,token=2)]
        x=self.oracle([[1],[2]],nodes,[1,2],[1,2],remaining_after_root=1)
        self.assertEqual(x['L_actual'],1);self.assertEqual(x['reason'],'normal_termination')

    def test_eos_and_root_eos(self):
        nodes=[dict(parent=-1,depth=1,token=99),dict(parent=0,depth=2,token=2)]
        x=self.oracle([[99],[2]],nodes,[99],[99,2])
        self.assertEqual(x['L_actual'],1);self.assertEqual(x['L_raw'],1)
        x=self.oracle([[99],[2]],nodes,[],[99,2],root_token=99)
        self.assertEqual(x['L_actual'],0);self.assertEqual(x['reason'],'normal_termination')

    def test_commit_inconsistency(self):
        nodes=[dict(parent=-1,depth=1,token=1)]
        x=self.oracle([[1]],nodes,[1],[])
        self.assertEqual(x['reason'],'verification_commit_inconsistency_pending')

    def test_unverified_reference_rejected(self):
        with self.assertRaises(ValueError):
            self.oracle([[1]],[],[1],[],reference_verified=False)

    def test_invalid_tree_and_short_reference_rejected(self):
        with self.assertRaises(ValueError):
            self.oracle([[1]],[dict(parent=0,depth=1,token=1)],[1],[])
        with self.assertRaises(ValueError):self.oracle([[1],[2]],[],[1],[])


if __name__=='__main__':unittest.main()
