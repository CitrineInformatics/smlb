"""Pseudo-random numbers tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.
"""

import pytest

import smlb


def test_random_order_independence():
    """Tests whether pseudo-random numbers are independent of call order."""

    def bar(rng):
        return smlb.Random(rng).random.rand()

    def baz(rng):
        return smlb.Random(rng).random.rand()

    rng1, rng2 = smlb.Random(7459).random.split(2)
    x1 = bar(rng1)
    x2 = baz(rng2)

    z2 = baz(rng2)  # different order of calls
    z1 = bar(rng1)

    assert x1 == z1 and x2 == z2

