"""Noise model tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
Citrine Informatics 2019.
"""

import pytest

import numpy as np
import scipy as sp
import scipy.stats

import smlb

#################
#  NormalNoise  #
#################


def test_NoNoise():
    """Test constant noise."""

    with pytest.raises(smlb.InvalidParameterError):
        smlb.NoNoise()

    noise = smlb.NoNoise(rng=1).noise(3)
    assert (noise == [0, 0, 0]).all()

    noise = smlb.NoNoise(value=1, rng=1).noise((1, 2))
    assert (noise == [[1, 1]]).all()


def test_NormalNoise():
    """Test Gaussian noise."""

    # fail without specifying pseudo-random number generator seed
    with pytest.raises(smlb.InvalidParameterError):
        smlb.NormalNoise()

    # unit normal
    noise = smlb.NormalNoise(rng=1).noise(100)
    assert sp.stats.normaltest(noise)[1] > 0.05

    # same seed leads to identical noise
    noise2 = smlb.NormalNoise(rng=1).noise(100)
    assert (noise == noise2).all()

    # non-unit normal
    noise = smlb.NormalNoise(mean=10, stddev=0.5, rng=1).noise(100)
    assert sp.stats.normaltest(noise)[1] > 0.05


@pytest.fixture
def fixture_TabularData_ComputedLabels():
    """TabularData with ComputedLabels."""

    def _create_ds(size, labelf):
        class TabularDataComputedLabels(smlb.TabularData):
            def __init__(self, size, **kwargs):
                data = np.arange(1, size + 1, 1).reshape((size, 1))
                self._labelf = labelf
                super().__init__(data=data, **kwargs)

            def labels(self, indices):
                return self._labelf(self.samples())

            @property
            def is_labeled(self):
                return True

        return TabularDataComputedLabels(size=size)

    return _create_ds


def test_LabelNoise_NoNoise(fixture_TabularData_ComputedLabels):
    """Test LabelNoise with NoNoise."""

    data1 = fixture_TabularData_ComputedLabels(
        size=5, labelf=lambda arg: np.power(arg.flatten(), 2)
    )
    data2 = smlb.LabelNoise(noise=smlb.NoNoise(rng=1)).fit(data1).apply(data1)

    assert (data1.labels([0, 1, 2, 3, 4]) == [1, 4, 9, 16, 25]).all()  # no change to incoming data
    assert (data2.labels([0, 1, 2, 3, 4]) == [1, 4, 9, 16, 25]).all()


def test_LabelNoise_NormalNoise(fixture_TabularData_ComputedLabels):
    """Test LabelNoise with NormalNoise."""

    arange = np.arange(0, 100)
    data1 = fixture_TabularData_ComputedLabels(size=100, labelf=lambda arg: arg.flatten())
    data2 = smlb.LabelNoise(noise=smlb.NormalNoise(rng=1)).fit(data1).apply(data1)
    assert sp.stats.normaltest(data2.labels(arange) - arange)[1] > 0.05
    assert sp.stats.normaltest(data2.labels(arange))[1] < 0.05

    # repeated evaluation of labels will yield different values
    assert (data2.labels(arange) != data2.labels(arange)).any()
