from numbers import Number
from typing import Callable, Optional, Union, Sequence, List

import smlb
from smlb import (
    params,
    Data,
    Features,
    SupervisedLearner,
    TabularData,
)


class SelectFromTotalImportance(Features):
    """Select features based on summed importance weights."""

    def __init__(
        self,
        learner: SupervisedLearner,
        total_importance: float = 0.95,
        min_features: Optional[int] = None,
        estimator_getter: Union[str, Callable] = "_model",
        *args,
        **kwargs,
    ):
        """Initialize State.

        Parameters:
            learner: Learner that contains an estimator. The estimator must be able to provide either
                ``feature_importances_`` or ``coef_`` attributes after fitting that are used for
                feature selection.
            total_importance: Total feature importance weight to include, between 0 and 1.
                Feature importances are pulled from the ``learner`` and unit normalized before including them
                in decreasing order of importance until ``total_weight`` is met or exceeded.
            min_features: Lower bound on the number of selected features, regardless of total weight.
                Default is ``None``, i.e. no lower bound.
            estimator_getter: Name of learner attribute that returns an estimator or a function that returns
                the estimator given a learner. Default is the ``_model`` attribute.
        """

        self.learner = params.instance(learner, SupervisedLearner)
        self.total_importance = params.real(total_importance, from_=0, to=1)

        is_nneg_int = lambda x: params.integer(x, from_=0)
        self.min_features = params.optional_(min_features, is_nneg_int)

        is_callable = lambda x: params.callable(x, num_pos_or_kw=1)
        self.estimator_getter = params.any_(estimator_getter, is_callable, params.string)

        self._support: Optional[List[int]] = None

        super().__init__(*args, **kwargs)

    def fit(self, data: Data) -> "SelectFromTotalImportance":
        """Fit the model with input ``data``.

        Parameters:
            data: data to fit

        Returns:
            the instance itself
        """
        self.learner.fit(data)
        self._select_support()

        return self

    def apply(self, data: Data) -> TabularData:
        """Select features from the data.

        Parameters:
            data: data to select features from

        Returns:
            data with selected features
        """
        data = params.instance(data, Data)
        samples = params.real_matrix(data.samples())

        if self._support is None:
            raise smlb.NotFittedError(self.learner)

        selected = samples[:, self._support]

        return TabularData(selected, data.labels())

    @property
    def _estimator(self):
        """Returns the estimator from the learner using the provided getter."""
        if isinstance(self.estimator_getter, str):
            return getattr(self.learner, self.estimator_getter)
        elif isinstance(self.estimator_getter, Callable):
            return self.estimator_getter(self.learner)
        else:
            raise AttributeError(
                f"Unable to get estimator using {self.estimator_getter}."
                "Getter must be a string representing the learner's estimator attribute "
                "or a function that returns the estimator given a learner."
            )

    @property
    def _feature_importances(self) -> Sequence[Number]:
        """Returns unnormalized feature importances from the learner's estimator.

        Assumes the estimator has been fit.
        """
        estimator = self._estimator
        for attr in {"feature_importances_", "coef_"}:
            try:
                importances = getattr(estimator, attr)
                return params.sequence(importances, type_=Number)
            except AttributeError:
                pass  # attribute not found, try the next option

        raise AttributeError(
            f"Unable to get importances from estimator {estimator}. "
            "Estimator must provide either `feature_importances_` or `coef_`."
        )

    def _select_support(self):
        """Select importances until total weight and minimum number of features (if specified) are met.

        Assumes the estimator has been fit.
        """
        importances = self._feature_importances
        total_importance = sum(importances)
        normalized_importances = [i / total_importance for i in importances]
        indexed_importances = list(enumerate(normalized_importances))
        current_total = 0.0
        self._support = []
        # sort by decreasing importance, and select support indices
        # until total weight and min feature requirements are met
        for idx, importance in sorted(indexed_importances, key=lambda x: x[-1], reverse=True):
            if current_total >= self.total_importance:
                if self.min_features is None or len(self._support) >= self.min_features:
                    break
            self._support.append(idx)
            current_total += importance
        # sort selected indices to ensure relative feature order is preserved when applied to data
        self._support.sort()
