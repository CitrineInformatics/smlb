import numpy as np
import pytest

from smlb import Data, InvalidParameterError
from smlb.decomposition import PCASklearn


def test_transform(friedman_1979_data: Data):
    n_components = 3
    pca = PCASklearn(rng=0, n_components=n_components)
    pca.fit(friedman_1979_data)
    transformed = pca.apply(friedman_1979_data)
    assert transformed.samples().shape == (friedman_1979_data.num_samples, n_components)


def test_inverse_transform(friedman_1979_data: Data):
    # keep original number of dimensions, so inversion is lossless
    n_components = len(friedman_1979_data.samples()[0])
    pca = PCASklearn(rng=0, n_components=n_components)
    pca.fit(friedman_1979_data)
    transformed = pca.apply(friedman_1979_data)
    inverted = pca.inverse().apply(transformed)

    actual = inverted.samples()
    expected = friedman_1979_data.samples()
    assert np.allclose(actual, expected)


def test_non_default_initialization():
    svd_solver = 'randomized'
    tol = 1.0
    iterated_power = 0
    pca = PCASklearn(
        svd_solver='randomized',
        tol=1.0,
        iterated_power=0
    )

    assert pca._pca.svd_solver == svd_solver
    assert pca._pca.tol == tol
    assert pca._pca.iterated_power == iterated_power


def test_invalid_initialization():
    with pytest.raises(InvalidParameterError):
        PCASklearn(n_components=0)

    with pytest.raises(InvalidParameterError):
        PCASklearn(svd_solver='foo')

    with pytest.raises(InvalidParameterError):
        PCASklearn(tol=-1)

    with pytest.raises(InvalidParameterError):
        PCASklearn(iterated_power=-1)
