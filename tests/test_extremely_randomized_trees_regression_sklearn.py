"""ExtraRandomTreesRegressionSklearn tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp 2020, Citrine Informatics.
"""

import pytest

import numpy as np

skl = pytest.importorskip("sklearn")

import smlb
from learners.scikit_learn.extremely_randomized_trees_regression_sklearn import (
    ExtremelyRandomizedTreesRegressionSklearn,
)


@pytest.mark.timeout(2)
def test_ExtremelyRandomizedTreesRegressionSklearn_1():
    """Simple examples."""

    # constant function
    # MH: for constant labels, expected uncertainties are zero
    train_data = smlb.TabularData(
        data=np.array([[-4], [-3], [-2], [-1], [0], [1], [2], [3], [4]]),
        labels=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    valid_data = smlb.TabularData(data=np.array([[-4], [-2], [0], [3], [4]]))
    rf = ExtremelyRandomizedTreesRegressionSklearn(
        n_estimators=10, uncertainties="naive", random_state=0
    )
    preds = rf.fit(train_data).apply(valid_data)
    mean, stddev = preds.mean, preds.stddev

    assert np.allclose(mean, [1, 1, 1, 1, 1])
    assert np.allclose(stddev, [0, 0, 0, 0, 0])

    # delta distributions (zero standard deviation)
    rf = ExtremelyRandomizedTreesRegressionSklearn(
        n_estimators=10, uncertainties=None, random_state=0
    )
    preds = rf.fit(train_data).apply(valid_data)
    mean, stddev = preds.mean, preds.stddev

    assert np.allclose(mean, [1, 1, 1, 1, 1])
    assert np.allclose(stddev, [0, 0, 0, 0, 0])

    assert isinstance(preds, smlb.DeltaPredictiveDistribution)


@pytest.mark.timeout(2)
def test_ExtremelyRandomizedTreesRegressionSklearn_2():
    """Simple examples: linear 1-d function."""

    rf = ExtremelyRandomizedTreesRegressionSklearn(random_state=1, uncertainties="naive")
    train_data = smlb.TabularData(
        data=np.array([[-2], [-1.5], [-1], [-0.5], [0], [0.5], [1], [1.5], [2]]),
        labels=np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]),
    )
    rf.fit(train_data)

    mean = rf.apply(smlb.TabularData(data=np.array([[-1], [0], [1]]))).mean
    assert np.allclose(mean, [-1, 0, 1], atol=0.2)

    stddev = rf.apply(smlb.TabularData(data=np.array([[-2], [0], [2]]))).stddev
    assert stddev[0] > stddev[1] < stddev[2]

    # without uncertainties
    rf = ExtremelyRandomizedTreesRegressionSklearn(
        random_state=1
    )  # default for uncertainties is None
    rf.fit(train_data)

    preds = rf.apply(smlb.TabularData(data=np.array([[-1], [0], [1]])))
    assert np.allclose(preds.mean, [-1, 0, 1], atol=0.2)

    assert isinstance(preds, smlb.DeltaPredictiveDistribution)


@pytest.mark.timeout(2)
def test_ExtremelyRandomizedTreesRegressionSklearn_3():
    """Ensure predictions are identical independent of uncertainties method used."""

    rf1 = ExtremelyRandomizedTreesRegressionSklearn(random_state=1, uncertainties=None)
    rf2 = ExtremelyRandomizedTreesRegressionSklearn(random_state=1, uncertainties="naive")
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


@pytest.mark.timeout(2)
def test_ExtremelyRandomizedTreesRegressionSklearn_5():
    """Non-trivial test case, including standard deviation."""

    n, m, xlen = 150, 600, 10
    train_inputs = np.reshape(np.linspace(-xlen / 2, +xlen / 2, n), (n, 1))
    train_labels = (train_inputs * 2 + 1).flatten()
    train_data = smlb.TabularData(data=train_inputs, labels=train_labels)
    train_data = smlb.LabelNoise(noise=smlb.NormalNoise(rng=0)).fit(train_data).apply(train_data)

    valid_inputs = np.reshape(np.linspace(-xlen / 2, +xlen / 2, m), (m, 1))
    valid_labels = (valid_inputs * 2 + 1).flatten()
    valid_data = smlb.TabularData(data=valid_inputs, labels=valid_labels)
    valid_data = smlb.LabelNoise(noise=smlb.NormalNoise(rng=1)).fit(valid_data).apply(valid_data)

    # 12 trees meets minimal requirements for jackknife estimates
    rf = ExtremelyRandomizedTreesRegressionSklearn(random_state=0, uncertainties="naive")
    preds = rf.fit(train_data).apply(valid_data)
    mae = smlb.MeanAbsoluteError().evaluate(valid_data.labels(), preds)

    # for perfect predictions, expect MAE of 1.12943
    # (absolute difference between draws from two unit normal distributions)
    assert np.allclose(mae, 1.13, atol=0.25)
    assert np.allclose(np.mean(preds.stddev), 1, atol=0.25)
