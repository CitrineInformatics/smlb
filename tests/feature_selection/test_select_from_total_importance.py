import numpy as np

from smlb import Data
from smlb.feature_selection import SelectFromTotalImportance
from smlb.learners import RandomForestRegressionSklearn


def _count_expected_features(learner: RandomForestRegressionSklearn, total_importance: float) -> int:
    """Counts the number of features required to meet the total weight.

    Assumes the learner has been fit.
    """
    estimator = learner._model
    feature_importances = estimator.feature_importances_
    current_total = 0.0
    count = 0
    for importance in sorted(feature_importances, reverse=True):
        if current_total >= total_importance:
            return count
        current_total += importance
        count += 1

    raise ValueError(f'Unable to meet total weight {total_importance}. '
                     f'Sum of importances = {sum(feature_importances)}.')


def test_transform_default_getter(friedman_1979_data: Data):
    learner = RandomForestRegressionSklearn(rng=0)
    total_importance = 0.5
    selector = SelectFromTotalImportance(learner, total_importance=total_importance)
    selector.fit(friedman_1979_data)
    selected = selector.apply(friedman_1979_data)
    expected_num_features = _count_expected_features(learner, total_importance)
    assert selected.samples().shape == (friedman_1979_data.num_samples, expected_num_features)


def test_transform_callable_getter(friedman_1979_data: Data):
    learner = RandomForestRegressionSklearn(rng=0)
    total_importance = 0.5

    def _get_estimator(learner):
        return learner._model

    selector = SelectFromTotalImportance(learner, total_importance=total_importance, estimator_getter=_get_estimator)
    selector.fit(friedman_1979_data)
    selected = selector.apply(friedman_1979_data)
    expected_num_features = _count_expected_features(learner, total_importance)
    assert selected.samples().shape == (friedman_1979_data.num_samples, expected_num_features)


def test_select_none(friedman_1979_data: Data):
    learner = RandomForestRegressionSklearn(rng=0)
    total_importance = 0.0
    selector = SelectFromTotalImportance(learner, total_importance=total_importance)
    selector.fit(friedman_1979_data)
    selected = selector.apply(friedman_1979_data)
    assert selected.samples().shape == (friedman_1979_data.num_samples, 0)


def test_select_all(friedman_1979_data: Data):
    learner = RandomForestRegressionSklearn(rng=0)
    total_importance = 1.0
    selector = SelectFromTotalImportance(learner, total_importance=total_importance)
    selector.fit(friedman_1979_data)
    selected = selector.apply(friedman_1979_data)
    assert np.allclose(selected.samples(), friedman_1979_data.samples())


def test_select_min_features(friedman_1979_data: Data):
    learner = RandomForestRegressionSklearn(rng=0)
    total_importance = 0.0
    min_features = 3
    selector = SelectFromTotalImportance(learner, total_importance=total_importance, min_features=min_features)
    selector.fit(friedman_1979_data)
    selected = selector.apply(friedman_1979_data)
    assert selected.samples().shape == (friedman_1979_data.num_samples, min_features)
