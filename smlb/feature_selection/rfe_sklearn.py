from typing import Callable, Optional, Union

from sklearn.feature_selection import RFE

from smlb import (
    params,
    SupervisedLearner,
    InvalidParameterError,
)
from smlb.feature_selection.feature_selector_sklearn import FeatureSelectorSklearn


class RFESklearn(FeatureSelectorSklearn):
    """Select features by recursive feature elimination, scikit-learn implementation.

    .. seealso::
        See `here <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE>`_ for full documentation.
    """

    def __init__(
        self,
        learner: SupervisedLearner,
        n_features_to_select: Optional[Union[int, float]] = None,
        step: Union[int, float] = 1,
        verbose: int = 0,
        estimator_getter: Union[str, Callable] = "_model",
        *args,
        **kwargs
    ):
        """Initialize State.

        Parameters:
            learner: Learner that contains an estimator. The estimator must be able to provide either
                ``feature_importances_`` or ``coef_`` attributes after fitting that are used for
                feature selection.
            n_features_to_select: number of features to select.
                If None, half of the features are selected.
                If integer, the parameter is the absolute number of features to select.
                If float between 0 and 1, it is the fraction of features to select.
            step: If greater than or equal to 1, then step corresponds to the (integer) number of features
                to remove at each iteration. If within (0.0, 1.0), then step corresponds to the percentage
                (rounded down) of features to remove at each iteration.
            verbose: Controls verbosity of output.
                If >0, current number of features will be printed during each iteration.
            estimator_getter: Name of learner attribute that returns an estimator or a function that returns
                the estimator given a learner. Default is the ``_model`` attribute.
        """

        learner = params.instance(learner, SupervisedLearner)

        is_nneg_int = lambda x: params.integer(x, from_=0)

        # We need to differentiate between int and float here because params
        # will cast the argument to an int when it tests is_nneg_int.
        # If you give it a float on the range (0.0, 1.0), you'll be left with
        # n_features == 0 after the conversion instead of the expected fraction of features.
        if isinstance(n_features_to_select, int):
            n_features_to_select = params.optional_(n_features_to_select, is_nneg_int)
        elif isinstance(n_features_to_select, float):
            is_valid_n_features_real = lambda x: params.real(x, above=0, to=1.0)
            n_features_to_select = params.optional_(n_features_to_select, is_valid_n_features_real)
        else:
            raise InvalidParameterError("integer or float", n_features_to_select)

        is_valid_step_int = lambda x: params.integer(x, from_=1)
        is_valid_step_real = lambda x: params.real(x, above=0.0, below=1.0)
        step = params.any_(step, is_valid_step_int, is_valid_step_real)

        verbose = is_nneg_int(verbose)

        is_callable = lambda arg: params.callable(arg, num_pos_or_kw=1)
        estimator_getter = params.any_(estimator_getter, params.string, is_callable)

        if isinstance(estimator_getter, str):
            estimator = getattr(learner, estimator_getter)
        else:
            estimator = estimator_getter(learner)

        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=step,
            verbose=verbose,
            importance_getter="auto",
        )

        super().__init__(selector=selector, *args, **kwargs)
