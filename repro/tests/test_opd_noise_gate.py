"""Noise allowance cannot override global or presence constraints."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'diagnostics'))
import unittest
from opd_noise_candidates import noise_gate


class NoiseGateTest(unittest.TestCase):
    def test_noise_does_not_override_global_failure(self):
        report=dict(cosine=.9998,relative_l2=.015,norm_relative=0.,
                    tensors=[dict(name='a',max_abs=.01,passed=False)])
        cal=dict(tensors=[dict(name='a',proposed_noise_threshold=.1)])
        result=noise_gate(report,cal)
        self.assertTrue(result['tensors'][0]['passed'])
        self.assertFalse(result['passed'])

    def test_presence_mismatch_is_not_noise(self):
        report=dict(cosine=1.,relative_l2=0.,norm_relative=0.,tensors=[dict(name='a',passed=False)])
        cal=dict(tensors=[dict(name='a',proposed_noise_threshold=100.)])
        self.assertFalse(noise_gate(report,cal)['passed'])

    def test_frozen_absolute_boundary(self):
        cal=dict(tensors=[dict(name='a',proposed_noise_threshold=.01)])
        report=dict(cosine=1.,relative_l2=0.,norm_relative=0.,tensors=[dict(name='a',max_abs=.01,passed=False)])
        self.assertTrue(noise_gate(report,cal)['passed'])
        report['tensors'][0]['max_abs']=.010001
        self.assertFalse(noise_gate(report,cal)['passed'])

if __name__=='__main__':unittest.main()
