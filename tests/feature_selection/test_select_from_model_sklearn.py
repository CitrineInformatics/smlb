from smlb import Data

from smlb.learners import RandomForestRegressionSklearn
from smlb.feature_selection import SelectFromModelSklearn


def test_transform_default_getter(friedman_1979_data: Data):
    estimator = RandomForestRegressionSklearn(rng=0)
    max_features = 2
    select_from_model = SelectFromModelSklearn(estimator, max_features=max_features)
    select_from_model.fit(friedman_1979_data)
    selected = select_from_model.apply(friedman_1979_data)
    assert selected.samples().shape == (friedman_1979_data.num_samples, max_features)


def test_transform_callable_getter(friedman_1979_data: Data):
    learner = RandomForestRegressionSklearn(rng=0)
    max_features = 2

    def _get_estimator(learner):
        return learner._model

    select_from_model = SelectFromModelSklearn(learner, estimator_getter=_get_estimator, max_features=max_features)
    select_from_model.fit(friedman_1979_data)
    selected = select_from_model.apply(friedman_1979_data)
    assert selected.samples().shape == (friedman_1979_data.num_samples, max_features)
