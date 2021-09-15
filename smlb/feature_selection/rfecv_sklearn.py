from typing import Callable, Optional, Union

from sklearn.feature_selection import RFECV

from smlb import (
    params,
    SupervisedLearner,
    InvalidParameterError,
)
from smlb.feature_selection.feature_selector_sklearn import FeatureSelectorSklearn


class RFECVSklearn(FeatureSelectorSklearn):
    """Select features by recursive feature elimination and cross-validated selection of the best number of features, scikit-learn implementation.

    .. seealso::
        See `here <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV>`_ for full documentation.
    """

    def __init__(
        self,
        learner: SupervisedLearner,
        step: Union[int, float] = 1,
        min_features_to_select: int = 1,
        cv: Optional[int] = None,
        scoring: Optional[Union[str, Callable]] = None,
        verbose: int = 0,
        n_jobs: Optional[int] = None,
        estimator_getter: Union[str, Callable] = "_model",
        *args,
        **kwargs
    ):
        """Initialize State.

        Parameters:
            learner: Learner that contains an estimator. The estimator must be able to provide either
                ``feature_importances_`` or ``coef_`` attributes after fitting that are used for
                feature selection.
            step: If greater than or equal to 1, then step corresponds to the (integer) number of features
                to remove at each iteration. If within (0.0, 1.0), then step corresponds to the percentage
                (rounded down) of features to remove at each iteration.
            min_features_to_select: The minimum number of features to be selected.
                This number of features will always be scored, even if the difference
                between the original feature count and min_features_to_select isn’t divisible by step.
            cv: Optional integer that specifies the number of folds to use in k-fold cross validation.
                Must be at least 2.
                If None, 5 folds are used.
            scoring: A ``str`` that corresponds to a predefined scorer (see `here
                <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_ for all predefined options)
                or a function with signature ``scorer(estimator, X, y, sample_weights=None)`` that returns a single value.
                If None, the estimator's default scorer (if available) is used.
            verbose: Controls verbosity of output.
                If >0, current number of features will be printed during each iteration.
            n_jobs: Number of jobs to run in parallel.
                When evaluating a new feature to add or remove, cross-validation is performed in parallel over the folds.
                None (default) means 1 unless in a ``joblib.parallel_backend context``.
                -1 means use all processors.
            estimator_getter: Name of learner attribute that returns an estimator or a function that returns
                the estimator given a learner. Default is the ``_model`` attribute.
        """

        learner = params.instance(learner, SupervisedLearner)

        is_valid_step_int = lambda x: params.integer(x, from_=1)
        is_valid_step_real = lambda x: params.real(x, above=0.0, below=1.0)
        step = params.any_(step, is_valid_step_int, is_valid_step_real)

        min_features_to_select = params.integer(min_features_to_select, from_=1)

        is_valid_n_folds = lambda x: params.integer(x, from_=2)
        cv = params.optional_(cv, is_valid_n_folds)

        is_callable_scorer = lambda x: params.callable(x, num_pos_or_kw=4)
        is_str_or_callable = lambda x: params.any_(x, params.string, is_callable_scorer)
        scoring = params.optional_(scoring, is_str_or_callable)

        verbose = params.integer(verbose, from_=0)

        if n_jobs == 0:
            raise InvalidParameterError("-1 or an integer > 0", n_jobs)
        else:
            is_valid_n_jobs = lambda x: params.integer(x, from_=-1)
            n_jobs = params.optional_(n_jobs, is_valid_n_jobs)

        is_callable = lambda arg: params.callable(arg, num_pos_or_kw=1)
        estimator_getter = params.any_(estimator_getter, params.string, is_callable)

        if isinstance(estimator_getter, str):
            estimator = getattr(learner, estimator_getter)
        else:
            estimator = estimator_getter(learner)

        selector = RFECV(
            estimator=estimator,
            step=step,
            min_features_to_select=min_features_to_select,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
            importance_getter="auto",
        )

        super().__init__(selector=selector, *args, **kwargs)
