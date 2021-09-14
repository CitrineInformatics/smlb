from sklearn.metrics import max_error, make_scorer

from smlb import Data
from smlb.feature_selection import RFECVSklearn
from smlb.learners import RandomForestRegressionSklearn


def test_transform_default_init(friedman_1979_data: Data):
    learner = RandomForestRegressionSklearn(rng=0)

    select_from_model = RFECVSklearn(learner=learner)
    select_from_model.fit(friedman_1979_data)
    selected = select_from_model.apply(friedman_1979_data)
    # friedman function has 5 informational dimensions
    assert selected.samples().shape == (friedman_1979_data.num_samples, 5)


def test_transform_callable_getter(friedman_1979_data: Data):
    learner = RandomForestRegressionSklearn(rng=0)
    min_features_to_select = 2
    cv = 2
    scoring = make_scorer(max_error)
    step = 0.2  # remove 20% of features at a time

    def _get_estimator(learner):
        return learner._model

    select_from_model = RFECVSklearn(
        learner=learner,
        step=step,
        min_features_to_select=min_features_to_select,
        cv=cv,
        scoring=scoring,
        estimator_getter=_get_estimator
    )
    select_from_model.fit(friedman_1979_data)
    selected = select_from_model.apply(friedman_1979_data)
    assert selected.samples().shape == (friedman_1979_data.num_samples, min_features_to_select)
