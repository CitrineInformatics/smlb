"""RandomForestRegressionSklearn tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp 2020, Citrine Informatics.
"""

import pytest

import numpy as np
from sklearn.metrics import pairwise_distances

skl = pytest.importorskip("sklearn")

import smlb
from learners.scikit_learn.random_forest_regression_sklearn import RandomForestRegressionSklearn

# todo: rework the example to be meaningful for random forests


def test_RandomForestRegressionSklearn_1():
    """Simple example: constant 1-d function."""

    # MH: for constant labels, expected uncertainties are zero
    train_data = smlb.TabularData(
        data=np.array([[-4], [-3], [-2], [-1], [0], [1], [2], [3], [4]]),
        labels=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    valid_data = smlb.TabularData(data=np.array([[-4], [-2], [0], [3], [4]]))
    rf = RandomForestRegressionSklearn(random_state=1, uncertainties="naive")
    preds = rf.fit(train_data).apply(valid_data)
    mean, stddev = preds.mean, preds.stddev

    assert np.allclose(mean, [1, 1, 1, 1, 1])
    assert np.allclose(stddev, [0, 0, 0, 0, 0])


def test_RandomForestRegressionSklearn_2():
    """Simple examples: linear 1-d function."""

    rf = RandomForestRegressionSklearn(random_state=1, uncertainties="naive", correlations="naive")
    train_data = smlb.TabularData(
        data=np.array([[-2], [-1.5], [-1], [-0.5], [0], [0.5], [1], [1.5], [2]]),
        labels=np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]),
    )
    rf.fit(train_data)

    mean = rf.apply(smlb.TabularData(data=np.array([[-1], [0], [1]]))).mean
    assert np.allclose(mean, [-1, 0, 1], atol=0.2)

    stddev = rf.apply(smlb.TabularData(data=np.array([[-2], [0], [2]]))).stddev
    assert stddev[0] > stddev[1] < stddev[2]

    corr = rf.apply(smlb.TabularData(data=np.array([[-1], [0], [1]])), compute_corr=True).corr
    assert abs(corr[0,1]) > abs(corr[0,2]) < abs(corr[1,2])

    # Confirm that the kernel behaves properly
    lower_indices = np.tril_indices(corr.shape[0], k=-1, m=corr.shape[1])
    correlations = np.abs(corr[lower_indices])
    absolute_differences = pairwise_distances([[-1], [0], [1]])[lower_indices]
    kernel_quality_metric = abs(np.corrcoef(correlations, absolute_differences)[0,1])
    assert kernel_quality_metric > 0

    # without uncertainties
    rf = RandomForestRegressionSklearn(random_state=1)  # default for uncertainties is None
    rf.fit(train_data)

    preds = rf.apply(smlb.TabularData(data=np.array([[-1], [0], [1]])))
    assert np.allclose(preds.mean, [-1, 0, 1], atol=0.2)

    assert isinstance(preds, smlb.DeltaPredictiveDistribution)


def test_RandomForestRegressionSklearn_3():
    """Ensure predictions are identical independent of uncertainties method used."""

    rf1 = RandomForestRegressionSklearn(random_state=1, uncertainties=None)
    rf2 = RandomForestRegressionSklearn(random_state=1, uncertainties="naive")
    train_data = smlb.TabularData(
        data=np.array([[-2], [-1.5], [-1], [-0.5], [0], [0.5], [1], [1.5], [2]]),
        labels=np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]),
    )
    rf1.fit(train_data)
    rf2.fit(train_data)

    test_data = np.array([[-3], [-1], [0], [0.5], [1], [2]])
    mean1 = rf1.apply(smlb.TabularData(data=test_data)).mean
    mean2 = rf2.apply(smlb.TabularData(data=test_data)).mean
    assert np.allclose(mean1, mean2, atol=1e-6)
