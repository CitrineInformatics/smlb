from typing import Callable, Optional, Union

from sklearn.feature_selection import SequentialFeatureSelector

from smlb import (
    params,
    SupervisedLearner,
    InvalidParameterError,
)
from smlb.feature_selection.feature_selector_sklearn import FeatureSelectorSklearn


class SequentialFeatureSelectorSklearn(FeatureSelectorSklearn):
    """Greedily select features sequentially (either forward or backward), scikit-learn implementation.

    .. seealso::
        See `here <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html#sklearn.feature_selection.SequentialFeatureSelector`_ for full documentation.
    """

    def __init__(
            self,
            learner: SupervisedLearner,
            n_features_to_select: Optional[Union[int, float]] = None,
            direction: str = 'forward',
            scoring: Optional[Union[str, Callable]] = None,
            cv: Optional[int] = None,
            n_jobs: Optional[int] = None,
            estimator_getter: Union[str, Callable] = "_model",
            *args,
            **kwargs
    ):
        """Initialize State.

        Parameters:
            learner: Learner that contains an estimator.
            n_features_to_select: number of features to select.
                If None, half of the features are selected.
                If integer, the parameter is the absolute number of features to select.
                If float between 0 and 1, it is the fraction of features to select.
            direction: Either ``'forward'`` or ``'backward'``,
                specifies whether to perform forward or backward selection.
                Default is ``'forward'``.
            scoring: A ``str`` that corresponds to a predefined scorer (see `here
                <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_ for all predefined options)
                or a function with signature ``scorer(estimator, X, y, sample_weights=None)`` that returns a single value.
                If None, the estimator's default scorer (if available) is used.
            cv: Optional integer that specifies the number of folds to use in k-fold cross validation.
                Must be at least 2.
                If None, 5 folds are used.
            n_jobs: Number of jobs to run in parallel.
                When evaluating a new feature to add or remove, cross-validation is performed in parallel over the folds.
                None (default) means 1 unless in a ``joblib.parallel_backend context``.
                -1 means use all processors.
            estimator_getter: Name of learner attribute that returns an estimator or a function that returns
                the estimator given a learner. Default is the ``_model`` attribute.
        """

        learner = params.instance(learner, SupervisedLearner)

        # We need to differentiate between int and float here because params
        # will cast the argument to an int when it tests the parameter.
        # 1 is a valid int and float but has a different meaning based on type.
        # If it's an int, it means select 1 feature.
        # If it's a float, it means select all features.
        if n_features_to_select is None:
            pass  # None is allowed, no need for further type checking
        elif isinstance(n_features_to_select, int):
            n_features_to_select = params.integer(n_features_to_select, from_=1)
        elif isinstance(n_features_to_select, float):
            n_features_to_select = params.real(n_features_to_select, above=0.0, to=1.0)
        else:
            raise InvalidParameterError("optional integer or float", n_features_to_select)

        direction = params.enumeration(direction, {'forward', 'backward'})

        is_callable_scorer = lambda x: params.callable(x, num_pos_or_kw=4)
        is_str_or_callable = lambda x: params.any_(x, params.string, is_callable_scorer)
        scoring = params.optional_(scoring, is_str_or_callable)

        is_valid_n_folds = lambda x: params.integer(x, from_=2)
        cv = params.optional_(cv, is_valid_n_folds)

        if n_jobs == 0:
            raise InvalidParameterError("-1 or an integer > 0", n_jobs)
        else:
            is_valid_n_jobs = lambda x: params.integer(x, from_=-1)
            n_jobs = params.optional_(n_jobs, is_valid_n_jobs)

        is_callable_getter = lambda arg: params.callable(arg, num_pos_or_kw=1)
        estimator_getter = params.any_(estimator_getter, params.string, is_callable_getter)

        if isinstance(estimator_getter, str):
            estimator = getattr(learner, estimator_getter)
        else:
            estimator = estimator_getter(learner)

        selector = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            direction=direction,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs
        )

        super().__init__(selector=selector, *args, **kwargs)
