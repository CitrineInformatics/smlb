"""Distributions tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.
"""

import pytest

import smlb

###############
#  Interface  #
###############


def test_interface_erroneous_arguments():
    """Tests whether errors are raised for wrong keyword arguments."""

    # PredictiveDistribution can not be instantiated because it is abstract

    # this test also ensures that PredictiveDistribution.__init__, if defined, calls super().__init__
    with pytest.raises(Exception):
        smlb.DeltaPredictiveDistribution([1, 2, 3], badarg=None)  # spelling error


def test_interface_decomposition():

    d = smlb.NormalPredictiveDistribution([1, 2, 3], [0.5, 0.5, 1])
    assert not (d.has_noise_part or d.has_signal_part)

    dd = smlb.DeltaPredictiveDistribution([1, 2, 3], noise_part=d)
    assert dd.has_noise_part and not dd.has_signal_part

    dd = smlb.DeltaPredictiveDistribution([1, 2, 3], signal_part=d)
    assert dd.has_signal_part and not dd.has_noise_part

    dd = smlb.DeltaPredictiveDistribution([1, 2, 3], noise_part=d, signal_part=d)
    assert dd.has_signal_part and dd.has_noise_part

    with pytest.raises(smlb.BenchmarkError):
        d.noise_part
    with pytest.raises(smlb.BenchmarkError):
        d.signal_part

