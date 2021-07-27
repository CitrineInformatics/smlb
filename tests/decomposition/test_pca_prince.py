import numpy as np

from smlb import Data
from smlb.decomposition import PCAPrince


def test_transform(friedman_1979_data: Data):
    n_components = 3
    pca = PCAPrince(rng=0, n_components=n_components)
    pca.fit(friedman_1979_data)
    transformed = pca.apply(friedman_1979_data)
    assert transformed.samples().shape == (friedman_1979_data.num_samples, n_components)


def test_inverse_transform(friedman_1979_data: Data):
    # keep original number of dimensions, so inversion is lossless
    n_components = len(friedman_1979_data.samples()[0])
    pca = PCAPrince(rng=0, n_components=n_components)
    pca.fit(friedman_1979_data)
    transformed = pca.apply(friedman_1979_data)
    inverted = pca.inverse().apply(transformed)

    actual = inverted.samples()
    expected = friedman_1979_data.samples()
    assert np.allclose(actual, expected)
