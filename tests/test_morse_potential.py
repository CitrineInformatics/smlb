"""Morse potential dataset tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.
"""

import pytest

import numpy as np

import smlb


def test_morse_potential_examples():
    """Tests instantiating Morse potential datasets."""

    from datasets.synthetic.morse_potential.morse_potential import MorsePotentialData

    (D, r0, a) = (1, 2, 3)
    mp = MorsePotentialData(D=D, r0=r0, a=a)

    assert mp.labels([[r0]]) == -1 * D
    assert (mp.D, mp.r0, mp.a) == (D, r0, a)
