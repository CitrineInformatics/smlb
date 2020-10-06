"""Friedman 1979 dataset tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2020, Citrine Informatics.
"""

import pytest

import numpy as np

import smlb


def test_friedman_1979_examples():
    """Tests instantiating and evaluating Friedman (1979) datasets."""

    from smlb.datasets.synthetic.friedman_1979.friedman_1979 import Friedman1979Data

    fd = Friedman1979Data()
    f5 = Friedman1979Data(dimensions=5)
    f9 = Friedman1979Data(dimensions=9)

    assert (fd.dimensions, f5.dimensions, f9.dimensions) == (6, 5, 9)

    # results from Mathematica reference implementation
    inp = np.asfarray(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6],
        ]
    )
    out = np.asfarray([7.927905195293134, 9.17785252292473])
    assert (
        (fd.labels(inp[:, :6]) == out).all()
        and (f5.labels(inp[:, :5]) == out).all()
        and (f9.labels(inp[:, :9]) == out).all()
    )

    # invalid inputs
    with pytest.raises(smlb.InvalidParameterError):
        fd.labels(inp)
