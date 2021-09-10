import pytest

import numpy as np

skl = pytest.importorskip("sklearn")

import smlb
from smlb.learners import LassoSklearn


def test_constant_1d():
    """Simple example: constant 1-d function."""

    train_data = smlb.TabularData(
        data=np.array([[-4], [-3], [-2], [-1], [0], [1], [2], [3], [4]]),
        labels=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    valid_data = smlb.TabularData(data=np.array([[-4], [-2], [0], [3], [4]]))
    lasso = LassoSklearn(rng=1)
    preds = lasso.fit(train_data).apply(valid_data)

    assert np.allclose(preds.mean, np.ones(valid_data.num_samples))
    assert np.allclose(preds.stddev, np.zeros(valid_data.num_samples))
    assert np.allclose(preds.corr, np.eye(valid_data.num_samples))


def test_linear_1d():
    """Simple examples: linear 1-d function."""

    lasso = LassoSklearn(rng=1, alpha=0.1)
    train_data = smlb.TabularData(
        data=np.array([[-2], [-1.5], [-1], [-0.5], [0], [0.5], [1], [1.5], [2]]),
        labels=np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]),
    )
    lasso.fit(train_data)

    valid_data = smlb.TabularData(data=np.array([[-1], [0], [1]]))
    preds = lasso.apply(valid_data)
    assert np.allclose(preds.mean, [-1, 0, 1], atol=0.2)
    assert np.allclose(preds.stddev, np.zeros(valid_data.num_samples))
    assert np.allclose(preds.corr, np.eye(valid_data.num_samples))
