"""Scientific data tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.
"""

import pytest

import numpy as np

import smlb

##################
#  element_data  #
##################


def test_element_data():
    """Tests chemical elements data."""

    # simple examples
    assert smlb.element_data(1, "abbreviation") == "H"

    # verify proton number
    assert all(smlb.element_data(i, "Z") == i for i in range(1, 119))

    # verify round trip
    abrvs = [smlb.element_data(i, "abbreviation") for i in range(1, 119)]
    np.random.seed(0)
    np.random.shuffle(abrvs)
    assert abrvs == [smlb.element_data(smlb.element_data(a, "Z"), "abbreviation") for a in abrvs]

