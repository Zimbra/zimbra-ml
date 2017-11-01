"""
Portions (C) Copyright 2017 Synacor, Inc.
Created: 9/22/2017, Michael Toutonghi

This and other files in this directory are fixes for the Neon library, typically in the form of a subclass, sometimes
a replacement or even a patch.

Some portions of this file are copyrighted by Intel and used without usage restriction in accordance
with the Apache 2.0 License, under which Intel Nervana Neon is licensed:
https://github.com/NervanaSystems/neon/blob/master/LICENSE
"""

from scipy.special import expit
from neon.backends.nervanacpu import numpy_call_dict_cpu

def fix_logistic_cpu(be):
    assert not numpy_call_dict.get('sig', None) is None
    numpy_call_dict['sig'] = lambda left: expit(left)

def fix_logistic(be):
    # TODO: we need to support GPU and sig2, which also numerically unstable
    fix_logistic_cpu(be)