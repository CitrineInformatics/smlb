"""Utility tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.
"""

import pytest

import numpy as np

import smlb

#################
#  is_sequence  #
#################


def test_is_sequence_examples():
    """Tests whether is_sequence complies to docstring via examples."""

    assert smlb.is_sequence([1, 2, 3]), "list"
    assert smlb.is_sequence((1, 2, 3)), "tuple"
    assert smlb.is_sequence(np.asfarray([1, 2, 3])), "array"

    assert not smlb.is_sequence("str"), "string"
    assert not smlb.is_sequence(b"bytes"), "bytes"
    assert not smlb.is_sequence(dict(a=1, b=2)), "dictionary"
    assert not smlb.is_sequence({1, 2, 3}), "set"
