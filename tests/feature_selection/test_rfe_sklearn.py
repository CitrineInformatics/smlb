from smlb import Data
from smlb.feature_selection import RFESklearn
from smlb.learners import RandomForestRegressionSklearn


def test_transform_default_getter(friedman_1979_data: Data):
    learner = RandomForestRegressionSklearn(rng=0)
    n_features_to_select = 4
    step = 2  # remove 2 features at a time

    select_from_model = RFESklearn(
        learner=learner,
        n_features_to_select=n_features_to_select,
        step=step
    )
    select_from_model.fit(friedman_1979_data)
    selected = select_from_model.apply(friedman_1979_data)
    assert selected.samples().shape == (friedman_1979_data.num_samples, n_features_to_select)


def test_transform_callable_getter(friedman_1979_data: Data):
    learner = RandomForestRegressionSklearn(rng=0)
    n_features_to_select = 0.5  # keep 50% of features (rounded down), so we should be left with 2 for friedman data
    step = 0.2  # remove 20% of features at a time

    def _get_estimator(learner):
        return learner._model

    select_from_model = RFESklearn(
        learner=learner,
        n_features_to_select=n_features_to_select,
        step=step,
        estimator_getter=_get_estimator
    )
    select_from_model.fit(friedman_1979_data)
    selected = select_from_model.apply(friedman_1979_data)
    assert selected.samples().shape == (friedman_1979_data.num_samples, 2)
