import unittest,copy
from protocol import *
class ProtocolTests(unittest.TestCase):
 def test_no_label_leak(self):
  r={'question':'Who is running?','gold_answer':'SECRET','options':['man','dog']}
  for m in ['A','B']:
   self.assertNotIn('SECRET',prompt(r,m));self.assertIn('A. man',prompt(r,m))
   rr={**r,'gold_answer':'different'};self.assertEqual(prompt(r,m),prompt(rr,m))
 def test_source(self):
  self.assertEqual(source_id('v_-D1gdv_gQyw'),source_id('-D1gdv_gQyw_1_20.avi'))
  self.assertEqual(source_id('bQJQGoJF7_k_162_169.avi'),'bQJQGoJF7_k')
  self.assertIsNone(source_id('4010069381.mp4'))
 def test_sampling(self):
  r=[{'video_id':str(v),'question_id':str(i),'question':f'Who number {i}?','question_type':('who','what')[i%2],'split':'val'} for v in range(25) for i in range(4)]
  a,q=select(r);self.assertEqual(len(a),20);self.assertEqual(len(q),40);self.assertEqual((a,q),select(r[::-1]));self.assertEqual({x['question_type'] for x in q},{'who','what'})
  with self.assertRaises(AssertionError):select([{**r[0],'split':'test'}])
 def test_cache_isolation(self):
  r={'hash':'x','sample_frame_ids':[1,2],'sample_timestamps':[.1,.2],'question':'what?'}
  k=cache_key(r,'A','p');self.assertNotEqual(k,cache_key(r,'B','p'));self.assertNotEqual(k,cache_key({**r,'question':'who?'},'A','p'));self.assertNotEqual(k,cache_key({**r,'sample_frame_ids':[2,1]},'A','p'))
 def test_exact_tokens(self):
  self.assertTrue(exact_tokens([1,2],[1,2]));self.assertFalse(exact_tokens([1,2],[1]));self.assertFalse(exact_tokens([1,2],[1,3]))
 def test_clipping(self):
  rr=[{'pos':100,'accepted':3},{'pos':104,'accepted':4}]
  c=clipped_rounds(rr,100,7);self.assertEqual(sum(x['new_root']+x['accepted_effective']+x['bonus_emitted'] for x in c),7);self.assertEqual(c[-1]['accepted_effective'],2)
  c=clipped_rounds(rr[:1],100,1);self.assertEqual(c[0]['accepted_effective'],0);self.assertEqual(c[0]['bonus_emitted'],0)
 def test_choice_parser(self):
  opts=['rest','run','swim','sit','eat']
  self.assertEqual(choice_answer('B. It is running.',opts),'B')
  self.assertEqual(choice_answer('Option C',opts),'C')
  self.assertEqual(choice_answer('C.',opts),'C')
  self.assertIsNone(choice_answer('A man is running.',opts))
  self.assertEqual(source_id('v_123456789_0_5.avi'),'v_123456789')
 def test_video_unit(self):
  a=video_average([{'video_id':'a','x':2},{'video_id':'a','x':4},{'video_id':'b','x':1}],'x');self.assertEqual(a,{'a':3,'b':1});p=paired(a,{'a':2,'b':0});self.assertEqual(p['mean'],1);self.assertEqual(p['n_videos'],2);self.assertEqual(p['ci_low'],1)
  with self.assertRaises(ValueError):paired(a,{'a':1})
if __name__=='__main__':unittest.main()
