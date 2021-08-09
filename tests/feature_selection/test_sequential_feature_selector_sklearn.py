from sklearn.metrics import max_error, make_scorer

from smlb import Data
from smlb.feature_selection import SequentialFeatureSelectorSklearn
from smlb.learners import RandomForestRegressionSklearn


def test_transform_default_init(friedman_1979_data: Data):
    learner = RandomForestRegressionSklearn(rng=0)

    sfs = SequentialFeatureSelectorSklearn(learner=learner)
    sfs.fit(friedman_1979_data)
    selected = sfs.apply(friedman_1979_data)
    # default n_features_to_select picks total_n_features // 2 (== 2 for friedman data)
    assert selected.samples().shape == (friedman_1979_data.num_samples, 2)


def test_transform_custom_init(friedman_1979_data: Data):
    learner = RandomForestRegressionSklearn(rng=0)
    n_features_to_select = 0.5  # keep 50% of features (rounded down), so we should be left with 2 for friedman data
    direction = 'backward'
    scoring = make_scorer(max_error)
    cv = 2
    n_jobs = 2

    def _get_estimator(learner):
        return learner._model

    select_from_model = SequentialFeatureSelectorSklearn(
        learner=learner,
        n_features_to_select=n_features_to_select,
        direction=direction,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        estimator_getter=_get_estimator
    )
    select_from_model.fit(friedman_1979_data)
    selected = select_from_model.apply(friedman_1979_data)
    assert selected.samples().shape == (friedman_1979_data.num_samples, 2)
