"""Hand-checkable QA contracts; run with python -m unittest discover ... ."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
import train_qa_opd as tr
from train_qa_sft import qa_blocks
import qa_anchors

class PositionTests(unittest.TestCase):
    def test_nonzero_anchor_all_accepted(self):
        # L=10, p=12, offset1 predicts answer3=23, not answer0=20.
        self.assertEqual(tr._verify([23,24,25],list(range(20,26)),12,10,6,None),(3,[True]*3))
    def test_zero_accept_keeps_suffix(self):
        self.assertEqual(tr._verify([99,24,25],list(range(20,26)),12,10,6,None),(0,[True]*3))
    def test_partial_accept_no_reaccept(self):
        self.assertEqual(tr._verify([23,99,25],list(range(20,26)),12,10,6,None),(1,[True]*3))
    def test_prompt_boundary(self):
        self.assertEqual(tr._verify([20,21],list(range(20,26)),9,10,6,None),(2,[True,True]))
    def test_reference_eos_inclusive(self):
        self.assertEqual(tr._verify([23,24,25],list(range(20,26)),12,10,6,4),(2,[True,True,False]))
    def test_tail_clipping(self):
        self.assertEqual(tr._verify([25,99,99],list(range(20,26)),14,10,6,None),(1,[True,False,False]))
    def test_proposal_eos_inclusive(self):
        self.assertEqual(tr._verify([23,99,25],list(range(20,26)),12,10,6,None,eos_token_ids=[99]),(1,[True,True,False]))
    def test_eos_first_reject(self):
        self.assertEqual(tr._verify([99,24,25],list(range(20,26)),12,10,6,None,eos_token_ids=99),(0,[True,False,False]))
    def test_no_valid(self):
        self.assertEqual(tr._verify([99,99],list(range(20,26)),15,10,6,5),(0,[False,False]))
    def test_verifier_contract_rejects_old_padding(self):
        with self.assertRaises(ValueError):tr._verify([1],[1,1],2,2,1,None)
    def test_sft_eos_label(self):
        full=torch.tensor([10,11,20,21,99]);cand=qa_anchors.sft_candidate_positions(2,2,3)
        _,labels,mask,_=qa_blocks(full,[3],torch.tensor([1]),4,0,cand[-1]+1,'cpu')
        self.assertEqual(labels[0,1].item(),99);self.assertEqual(mask[0].tolist(),[0,1,0,0])
    def test_teacher_root_row_and_draft_prefix(self):
        seen={}
        class Target:
            lm_head=torch.nn.Identity()
            def model(self,ids,**kw):
                seen.update(ids=ids.tolist(),positions=kw['cache_position'].tolist(),cache=kw['past_key_values'])
                # Output depends on all consumed proposal tokens, not reference suffix.
                return SimpleNamespace(last_hidden_state=ids.float().cumsum(1)[...,None])
        target=Target();target.lm_head.weight=torch.zeros(1)
        prep={'full':torch.tensor([10,11,12,13,14,15,16]),'prefix_kv':None,'vispos':lambda x:x}
        with patch.object(tr,'_cache_from_prefix',side_effect=lambda _,n:n):
            out=tr.teacher_proposal_logits(target,prep,3,[50,60,70],3,'cpu')
        self.assertEqual(seen,{'ids':[[13,50,60]],'positions':[3,4,5],'cache':3})
        self.assertEqual(out[:,0].tolist(),[13,63,123])
    def test_loss_formula_and_gradient(self):
        # Exactly invoke production _anchor_loss; independent direct tensor formula.
        q=torch.tensor([[1.,3.,0.],[2.,0.,1.],[0.,2.,1.]],requires_grad=True)
        teacher=torch.tensor([[0.,3.,1.],[0.,2.,1.],[2.,0.,1.]])
        prop=q.argmax(-1).tolist();full=torch.tensor([0,0,0,0,1,2,0])
        prep={'full':full,'prompt_len':3,'fused':torch.zeros(1,7,1),'slot':{'video_id':'test'}}
        for n,valid in [(1,[True]*3),(0,[True]*3),(3,[True]*3),(1,[True,True,False])]:
            # Force teacher verification to agree with the stipulated leading count.
            t=teacher.clone()
            for j in range(3):
                t[j]=torch.zeros(3);t[j,prop[j] if j<n else (prop[j]+1)%3]=3
            ai,ri=tr.opd_fullkl.stream_split(3,n,valid)
            plan={'pos':3,'prop':prop,'n_acc':n,'valid':valid,'acc_idx':ai,'rej_idx':ri}
            z=float(tr.gamma_weights(ri,.8,'cpu').sum())
            with patch.object(tr,'draft_block_logits_mixed',return_value=q),patch.object(tr,'teacher_proposal_logits',return_value=t[:max(ai.tolist()+ri.tolist())]):
                loss,stats=tr._anchor_loss(None,None,prep,plan,.8,'cpu',4,0,None,len(ai),z,bool(len(ai)),bool(len(ri)),torch.float32)
            terms=[]
            if len(ai):terms.append(torch.nn.functional.cross_entropy(q[ai-1],full[3+ai]))
            if len(ri):
                lq=q[ri-1].log_softmax(-1);lp=t[ri-1].log_softmax(-1)
                w=torch.tensor([.8**(int(k)-1) for k in ri])
                terms.append(((lq.exp()*(lq-lp)).sum(-1)*w).sum()/w.sum())
            ref=sum(terms)/len(terms)
            torch.testing.assert_close(loss,ref)
            torch.testing.assert_close(torch.autograd.grad(loss,q,retain_graph=True)[0],torch.autograd.grad(ref,q)[0])
            self.assertEqual(stats['online_n_acc'],n)

if __name__=='__main__':unittest.main()
