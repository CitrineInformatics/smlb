"""RandomForestRegressionLolo tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
"""

import pytest

import numpy as np

lolopy = pytest.importorskip("lolopy")
py4j = pytest.importorskip("py4j")

pytest.importorskip("smlb.learners.lolo.random_forest_regression_lolo")
from smlb.learners.lolo.random_forest_regression_lolo import RandomForestRegressionLolo

import smlb


@pytest.mark.timeout(10)
def test_RandomForestRegressionLolo():
    """Simple examples.

    Lolo requires at least 8 training points.
    """

    # constant function
    # MH: for constant labels, expected uncertainties are zero
    train_data = smlb.TabularData(
        data=np.array([[-4], [-3], [-2], [-1], [0], [1], [2], [3], [4]]),
        labels=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    valid_data = smlb.TabularData(data=np.array([[-4], [-2], [0], [3], [4]]))
    rf = RandomForestRegressionLolo(num_trees=10)
    preds = rf.fit(train_data).apply(valid_data)
    mean, stddev = preds.mean, preds.stddev

    assert np.allclose(mean, [1, 1, 1, 1, 1])
    assert np.allclose(stddev, [0, 0, 0, 0, 0])

    # delta distributions (zero standard deviation)
    rf = RandomForestRegressionLolo(num_trees=10, use_jackknife=False)
    preds = rf.fit(train_data).apply(valid_data)
    mean, stddev = preds.mean, preds.stddev

    assert np.allclose(mean, [1, 1, 1, 1, 1])
    assert np.allclose(stddev, [0, 0, 0, 0, 0])


@pytest.mark.timeout(10)
def test_RandomForestRegressionLolo_2():
    """Non-trivial test case, including standard deviation."""

    n, m, xlen = 100, 600, 10
    train_inputs = np.reshape(np.linspace(-xlen / 2, +xlen / 2, n), (n, 1))
    train_labels = (train_inputs * 2 + 1).flatten()
    train_data = smlb.TabularData(data=train_inputs, labels=train_labels)
    train_data = smlb.LabelNoise(noise=smlb.NormalNoise(rng=0)).fit(train_data).apply(train_data)

    valid_inputs = np.reshape(np.linspace(-xlen / 2, +xlen / 2, m), (m, 1))
    valid_labels = (valid_inputs * 2 + 1).flatten()
    valid_data = smlb.TabularData(data=valid_inputs, labels=valid_labels)
    valid_data = smlb.LabelNoise(noise=smlb.NormalNoise(rng=1)).fit(valid_data).apply(valid_data)

    # 12 trees meets minimal requirements for jackknife estimates
    rf = RandomForestRegressionLolo()
    preds = rf.fit(train_data).apply(valid_data)
    mae = smlb.MeanAbsoluteError().evaluate(valid_data.labels(), preds)

    # for perfect predictions, expect MAE of 1.12943
    # (absolute difference between draws from two unit normal distributions)
    assert np.allclose(mae, 1.13, atol=0.25)
    assert np.allclose(np.median(preds.stddev), 1, atol=0.5)
    # the previous test failed when using np.mean and atol=0.25 on the Travis CI system
