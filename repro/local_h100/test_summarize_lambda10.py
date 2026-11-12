import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from local_h100.summarize_lambda10 import validate_training_pair


def pair():
    base=dict(config=dict(e2e_tv_weight=0,init_from='/same',seed=12,updates=80),
              hashes=dict(initial_head_weights={'model.safetensors':'a'},schedule='s',runtime='r'),
              protocol_sha256='old')
    candidate=copy.deepcopy(base)
    candidate['config']['e2e_tv_weight']=10.
    candidate['protocol_sha256']='new'
    return base,candidate


def test_only_auxiliary_weight_may_change():
    validate_training_pair(*pair())


@pytest.mark.parametrize('section,key,value',[
    ('config','init_from','/trained_lambda0/step80'),
    ('config','seed',13),('config','updates',160),
    ('hashes','schedule','different'),('hashes','runtime','different')])
def test_confounding_change_is_rejected(section,key,value):
    base,candidate=pair();candidate[section][key]=value
    with pytest.raises(ValueError,match='beyond lambda'):
        validate_training_pair(base,candidate)
