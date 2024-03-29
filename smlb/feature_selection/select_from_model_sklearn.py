from typing import Callable, Optional, Union

from sklearn.feature_selection import SelectFromModel

from smlb import (
    params,
    SupervisedLearner,
)
from smlb.feature_selection.feature_selector_sklearn import FeatureSelectorSklearn


class SelectFromModelSklearn(FeatureSelectorSklearn):
    """Select features based on importance weights using the scikit-learn SelectFromModel implementation.

    .. seealso::
        See `here <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel>`_ for full documentation.
    """

    def __init__(
        self,
        learner: SupervisedLearner,
        threshold: Optional[Union[str, float]] = None,
        prefit: bool = False,
        norm_order: int = 1,
        max_features: Optional[int] = None,
        estimator_getter: Union[str, Callable] = "_model",
        *args,
        **kwargs
    ):
        """Initialize State.

        Parameters:
            learner: Learner that contains an estimator. The estimator must be able to provide either
                ``feature_importances_`` or ``coef_`` attributes after fitting that are used for
                feature selection.
            threshold: Threshold value to use for feature selection. Features whose
                importance is greater or equal to ``threshold`` are kept while the others are
                discarded. If "median" (resp. "mean"), then the ``threshold`` value is
                the median (resp. the mean) of the feature importances. A scaling
                factor (e.g., "1.25*mean") may also be used. If None and if the
                estimator has a parameter penalty set to l1, either explicitly
                or implicitly (e.g, Lasso), the threshold used is 1e-5.
                Otherwise, "mean" is used by default.
            prefit: Whether a prefit model is expected to be passed into the constructor directly or not.
            norm_order: Order of the norm used to filter the vectors of coefficients below
                ``threshold`` in the case where the ``coef_`` attribute of the estimator is of dimension 2.
                In other words, this is the order provided to ``np.linalg.norm`` to scale importances before
                computing the threshold and selecting features.
            max_features: The maximum number of features to select.
            estimator_getter: Name of learner attribute that returns an estimator or a function that returns
                the estimator given a learner. Default is the ``_model`` attribute.
        """

        learner = params.instance(learner, SupervisedLearner)

        is_str_or_float = lambda arg: params.any_(arg, params.string, params.real)
        threshold = params.optional_(threshold, is_str_or_float)

        prefit = params.boolean(prefit)
        norm_order = params.integer(norm_order, from_=1)
        max_features = params.optional_(max_features, params.integer)

        is_callable = lambda arg: params.callable(arg, num_pos_or_kw=1)
        estimator_getter = params.any_(estimator_getter, params.string, is_callable)

        if isinstance(estimator_getter, str):
            estimator = getattr(learner, estimator_getter)
        else:
            estimator = estimator_getter(learner)

        selector = SelectFromModel(
            estimator=estimator,
            threshold=threshold,
            prefit=prefit,
            norm_order=norm_order,
            max_features=max_features,
            importance_getter="auto",
        )

        super().__init__(selector=selector, *args, **kwargs)
