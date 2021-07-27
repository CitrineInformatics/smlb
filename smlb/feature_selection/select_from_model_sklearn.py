from typing import Callable, Optional, Union

from sklearn.feature_selection import SelectFromModel

from smlb import (
    params,
    Data,
    Features,
    SupervisedLearner,
    TabularData,
)


class SelectFromModelSklearn(Features):
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
                ``threshold`` in the case where the ``coef_`` attribute of the
                estimator is of dimension 2.
            max_features: The maximum number of features to select.
            estimator_getter: Name of learner attribute that returns an estimator or a function that returns
                the estimator given a learner. Default is the ``_model`` attribute.

        """

        learner = params.instance(learner, SupervisedLearner)

        is_str_or_float = lambda arg: params.any_(arg, params.string, params.real)
        threshold = params.optional_(threshold, is_str_or_float)

        prefit = params.boolean(prefit)
        norm_order = params.integer(norm_order)
        max_features = params.optional_(max_features, params.integer)

        is_callable = lambda arg: params.callable(arg, num_pos_or_kw=1)
        estimator_getter = params.any_(estimator_getter, params.string, is_callable)

        if isinstance(estimator_getter, str):
            estimator = getattr(learner, estimator_getter)
        else:
            estimator = estimator_getter(learner)

        super().__init__(*args, **kwargs)
        self._select_from_model = SelectFromModel(
            estimator=estimator,
            threshold=threshold,
            prefit=prefit,
            norm_order=norm_order,
            max_features=max_features,
            importance_getter="auto",
        )

    def fit(self, data: Data) -> "SelectFromModelSklearn":
        """Fit the model with input ``data``.

        Parameters:
            data: data to fit

        Returns:
            the instance itself
        """
        data = params.instance(data, Data)
        n = data.num_samples

        xtrain = params.real_matrix(data.samples(), nrows=n)
        ytrain = params.real_vector(data.labels(), dimensions=n)

        self._select_from_model.fit(xtrain, ytrain)

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

        support = self._select_from_model.get_support()
        selected = samples[:, support]

        return TabularData(selected, data.labels())
