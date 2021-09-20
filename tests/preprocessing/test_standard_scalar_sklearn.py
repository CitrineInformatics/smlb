import numpy as np

from smlb import Data
from smlb.preprocessing import StandardScalarSklearn


def test_transform(friedman_1979_data: Data):
    standard_scalar = StandardScalarSklearn()
    standard_scalar.fit(friedman_1979_data)
    transformed = standard_scalar.apply(friedman_1979_data)
    data = transformed.samples()
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # mean of each feature should be 0 after standardization
    assert np.allclose(mean, 0)
    # std of each feature should be 1 after standardization
    assert np.allclose(std, 1)


def test_inverse_transform(friedman_1979_data: Data):
    standard_scalar = StandardScalarSklearn()
    standard_scalar.fit(friedman_1979_data)
    transformed = standard_scalar.apply(friedman_1979_data)
    inverted = standard_scalar.inverse().apply(transformed)

    actual = inverted.samples()
    expected = friedman_1979_data.samples()
    # inverting standardized data should return original samples
    assert np.allclose(actual, expected)
