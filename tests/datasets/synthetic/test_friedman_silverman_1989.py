"""Friedman & Silverman 1989 dataset tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2020, Citrine Informatics.
"""

import pytest

import numpy as np

import smlb


def test_friedman_silverman_1989_examples():
    """Tests instantiating and evaluating Friedman (1979) datasets."""

    from smlb.datasets.synthetic.friedman_silverman_1989.friedman_silverman_1989 import (
        FriedmanSilverman1989Data,
    )

    fd = FriedmanSilverman1989Data()
    f5 = FriedmanSilverman1989Data(dimensions=5)
    f11 = FriedmanSilverman1989Data(dimensions=11)

    assert (fd.dimensions, f5.dimensions, f11.dimensions) == (10, 5, 11)

    # results from Mathematica reference implementation,
    # which was qualitatively verified by comaparing against Figure 4 in the original publication.
    inp = np.asfarray(
        [
            [0.25, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1],
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [
                0.11098378116007246,
                0.9316687928837954,
                0.11109168170199246,
                0.6687658063729789,
                0.10793285854914858,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        ]
    )
    out = np.asfarray([8.271828182845905, 2.359072962390666, 2.6157172979815355, 5.933910328413488])
    assert (
        # there appear to be differences in the last digit between Mathematica and NumPy
        np.allclose(fd.labels(inp[:, :10]), out, 0, 1e-15)
        and np.allclose(f5.labels(inp[:, :5]), out, 0, 1e-15)
        and np.allclose(f11.labels(inp[:, :11]), out, 0, 1e-15)
    )

    # invalid inputs
    with pytest.raises(smlb.InvalidParameterError):
        fd.labels(inp)
