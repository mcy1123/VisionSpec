import copy
import unittest
from argparse import Namespace
from unittest.mock import patch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
from opd_official_kl.parallel_trainer import validate_checkpoint, SLOT_MAPPING
from opd_official_kl.trainer import identity


class CheckpointContract(unittest.TestCase):
    def state(self):
        return dict(resume_schema=1, resume_identity={'x':1}, update=90,
                    update_step=90, data_cursor=90, diagnostic_only=False)

    def parallel(self):
        s=self.state()
        s['parallel_state']=dict(world_size=4, slot_mapping=SLOT_MAPPING,
            per_rank_rng=[{}, {}, {}, {}], implementation='rollout_rank0_serial_v1',
            schedule_cursor=90)
        return s

    def test_initialize_is_explicit(self):
        s=self.state()
        self.assertEqual(validate_checkpoint(s, {'x':1}, initialize=True),90)
        with self.assertRaisesRegex(ValueError,'topology'):
            validate_checkpoint(s, {'x':1})
        with self.assertRaisesRegex(ValueError,'original'):
            validate_checkpoint(self.parallel(), {'x':1}, initialize=True)

    def test_resume_rejects_topology_mapping_rng_cursor(self):
        s=self.parallel(); self.assertEqual(validate_checkpoint(s,{'x':1}),90)
        for key,value in [('world_size',2),('slot_mapping',{'0':[0,2]}),
                          ('per_rank_rng',[{}]),('schedule_cursor',91)]:
            bad=copy.deepcopy(s);bad['parallel_state'][key]=value
            with self.assertRaisesRegex(ValueError,'topology'):
                validate_checkpoint(bad,{'x':1})

    def test_diagnostic_cannot_be_promoted(self):
        s=self.parallel();s['diagnostic_only']=True
        with self.assertRaisesRegex(ValueError,'diagnostic'):
            validate_checkpoint(s,{'x':1})
        self.assertEqual(validate_checkpoint(s,{'x':1},diagnostic=True),90)
        with self.assertRaises(ValueError):
            validate_checkpoint(s,{'x':1},initialize=True,diagnostic=True)
        self.assertEqual(validate_checkpoint(self.state(),{'x':1},initialize=True,diagnostic=True),90)

    def test_recipe_and_requested_step(self):
        for kw in [dict(metadata={'x':2}),dict(metadata={'x':1},requested_step=91)]:
            with self.assertRaises(ValueError):validate_checkpoint(self.parallel(),**kw)

    def test_legacy_identity_is_unchanged(self):
        base=Namespace(schedule='s',pool='p',init_from='h',seed=7,out='o',gpu=0)
        new=copy.deepcopy(base);new.parallel_rollout=4;new.initialize_from='q';new.diagnostic_gate='gate'
        with patch('shared.training_resume.file_hash',return_value='sha'):
            self.assertEqual(identity(base),identity(new))


if __name__=='__main__':unittest.main()
