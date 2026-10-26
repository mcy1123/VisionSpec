"""CPU-only checks of new tail-batch transport and frozen paired schedule."""
import argparse
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from opd_official_kl.parallel_trainer import ParallelRollout
from opd_official_kl.native_contracts import global_denominator
from scienceqa_image.full_epoch.run import Controller,DEFAULT,DIAGNOSTIC


class QuickTests(unittest.TestCase):
    def test_partial_batch_rank0_receives_only_live_slot(self):
        obj=object.__new__(ParallelRollout);obj.rank=0;obj.metadata={'config_sha256':'x'}
        obj.head_sha='a'*64
        obj.barrier=lambda *a:{};obj.phase_check=lambda error: self.assertIsNone(error)
        obj.announce=lambda value:value
        rows=pd.DataFrame([dict(task='a',prompt='A'),dict(task='b',prompt='B')]).set_index('task')
        items=[dict(task='a',slot=0),dict(task='b',slot=1)]
        records=[dict(slot=i,task=t,response_count=1,ordered_rejected_offsets=[],denominator_hex=(1.).hex()) for i,t in enumerate(['a','b'])]
        obj.collect_slot=lambda *a:({'slot':0},records[0])
        obj.gather=lambda value:[records[0],records[1],None,None]
        with patch('opd_official_kl.parallel_trainer.receive_trace',return_value=({'slot':1},7)) as recv, \
             patch('opd_official_kl.parallel_trainer.validate_envelope'), \
             patch('opd_official_kl.parallel_trainer.denominator_record',side_effect=lambda t,s,k:records[s]), \
             patch('opd_official_kl.parallel_trainer.runtime.denominator',return_value=1.):
            traces,denom,*_=obj.collect(None,None,rows,items,None,0,0)
        self.assertEqual(set(traces),{0,1});self.assertEqual(denom,2.)
        recv.assert_called_once_with(1)

    def test_frozen_schedule_and_both_commands(self):
        c=Controller(argparse.Namespace(run=Path(DEFAULT),diagnostic_root=Path(DIAGNOSTIC),gpus='1,2,6,7'))
        rows=[json.loads(l) for l in c.s.read_text().splitlines()]
        self.assertEqual(len(rows),4726);self.assertEqual(len({r['task'] for r in rows}),4726)
        self.assertEqual([r['slot'] for r in rows[-2:]],[0,1]);self.assertEqual(rows[-1]['update'],1181)
        for arm in ['sft','opd']:
            cmd=c.command(arm,Path('/mnt/local/unused'),2)
            self.assertIn('--allow-partial-final',cmd)
            self.assertEqual(cmd[cmd.index('--updates')+1],'1182')
            self.assertEqual(cmd[cmd.index('--schedule')+1],str(c.s))
        manifest=pd.read_csv(c.run/'dataset/records_manifest.csv')
        sets=[set(g.pixel_sha256) for _,g in manifest.groupby('canonical_split')]
        for i,a in enumerate(sets):
            for b in sets[i+1:]:self.assertFalse(a&b)
        self.assertEqual(manifest[manifest.canonical_split=='test'].pixel_sha256.nunique(),200)

if __name__=='__main__':unittest.main()
