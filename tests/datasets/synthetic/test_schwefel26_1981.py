"""Schefel26 1981 dataset tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2020, Citrine Informatics.
"""

import pytest

import numpy as np

import smlb


def test_schwefel26_1981_examples():
    """Tests instantiating and evaluating Schwefel26 (1981) datasets."""

    from smlb.datasets.synthetic.schwefel26_1981.schwefel26_1981 import Schwefel261981Data

    s1 = Schwefel261981Data(dimensions=1)
    s2 = Schwefel261981Data(dimensions=2)
    s9 = Schwefel261981Data(dimensions=9)

    assert (s1.dimensions, s2.dimensions, s9.dimensions) == (1, 2, 9)

    # results from Mathematica reference implementation

    # minima
    inp = np.asfarray([[420.9687] * 9])
    assert np.allclose(s1.labels(inp[:,:1]), [0], atol=1e-4)
    assert np.allclose(s2.labels(inp[:,:2]), [0], atol=1e-4)
    assert np.allclose(s9.labels(inp[:,:9]), [0], atol=1e-3)
    
    inp = np.asfarray(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6],
        ]
    )
    assert np.allclose(s1.labels(inp[:,:1]), [418.9518016407093, 418.6580815304599], atol=1e-6)
    assert np.allclose(s2.labels(inp[:,:2]), [837.8482106729184, 837.4045306835739], atol=1e-6)
    assert np.allclose(s9.labels(inp[:,:9]), [3767.716410053263, 3767.716410053263], atol=1e-6)

    # invalid inputs
    with pytest.raises(smlb.InvalidParameterError):
        s2.labels(inp)
