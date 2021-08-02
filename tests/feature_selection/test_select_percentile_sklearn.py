import numpy as np

from smlb import Data, TabularData
from smlb.feature_selection import SelectPercentileSklearn
from smlb.learners import RandomForestRegressionSklearn


def _count_expected_features(estimator: RandomForestRegressionSklearn, percentile: int) -> int:
    """Count the number of features required to meet the specified percentile. Assumes the estimator has been fit."""
    feature_importances = estimator._model.feature_importances_
    threshold = np.percentile(feature_importances, 100 - percentile)
    return sum(feature_importances > threshold)


def test_select_from_feature_importances(friedman_1979_data: Data):
    percentile = 50
    estimator = RandomForestRegressionSklearn(rng=0)

    def score_func(X, y):
        data = TabularData(X, y)
        estimator.fit(data)
        return estimator._model.feature_importances_

    select_percentile = SelectPercentileSklearn(score_func, percentile=percentile)
    select_percentile.fit(friedman_1979_data)
    selected = select_percentile.apply(friedman_1979_data)
    # number of samples should not change
    assert selected.num_samples == friedman_1979_data.num_samples

    actual_num_features = selected.samples().shape[-1]
    expected_num_features = _count_expected_features(estimator, percentile)
    assert actual_num_features == expected_num_features
