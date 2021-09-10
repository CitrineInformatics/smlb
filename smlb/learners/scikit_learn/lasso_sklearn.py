from sklearn.linear_model import Lasso

from smlb import (
    Data,
    DeltaPredictiveDistribution,
    params,
    SupervisedLearner,
    Random,
)


class LassoSklearn(SupervisedLearner, Random):
    """Linear Model trained with L1 prior as regularizer (aka the Lasso), scikit-learn implementation.

    .. seealso::
        See `here <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso>`_ for full documentation.
    """

    def __init__(
        self,
        rng: int = None,
        alpha: float = 1.0,
        *,
        fit_intercept: bool = True,
        normalize: bool = False,
        precompute: bool = False,
        copy_X: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        warm_start: bool = False,
        positive: bool = False,
        selection: str = "cyclic",
        **kwargs,
    ):
        """Initialize state.

        Parameters:
            rng : Seed of the pseudo random number generator that selects a random
                feature to update.
                Used when ``selection`` == 'random'.
                Pass an int for reproducible output across multiple function calls.
                Default is None.
            alpha : Constant that multiplies the L1 term. Defaults to 1.0.
                ``alpha = 0`` is equivalent to an ordinary least square. For numerical
                reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
            fit_intercept : Whether to calculate the intercept for this model.
                If set to False, no intercept will be used in calculations
                (i.e. data is expected to be centered).
                Default is True.
            normalize : This parameter is ignored when ``fit_intercept`` is set to False.
                If True, the regressors X will be normalized before regression by
                subtracting the mean and dividing by the l2-norm.
                If you wish to standardize, please use
                :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
                on an estimator with ``normalize=False``.
                Default is False.
            precompute : Whether to use a precomputed Gram matrix to speed up calculations.
                The Gram matrix can also be passed as argument.
                For sparse input this option is always ``False`` to preserve sparsity.
                Default is False.
            copy_X : If ``True``, X will be copied; else, it may be overwritten.
                Default is True.
            max_iter : The maximum number of iterations.
                Default is 1000.
            tol : The tolerance for the optimization: if the updates are smaller than ``tol``,
                the optimization code checks the dual gap for optimality and continues until
                it is smaller than ``tol``.
                Default is 1e-4.
            warm_start : When set to True, reuse the solution of the previous call to fit as
                initialization, otherwise, just erase the previous solution.
                Default is False.
            positive : When set to ``True``, forces the coefficients to be positive.
                Default is False.
            selection : Either ``'cyclic'`` or ``'random'``.
                If set to ``'random'``, a random coefficient is updated every iteration
                rather than looping over features sequentially by default.
                This (setting to ``'random'``) often leads to significantly faster convergence
                especially when ``tol`` is higher than 1e-4.
                Default is ``'cyclic'``.
        """
        super().__init__(rng=rng, **kwargs)

        self._model = Lasso(
            alpha=params.real(alpha, from_=0, to=1),
            fit_intercept=params.boolean(fit_intercept),
            normalize=params.boolean(normalize),
            precompute=params.boolean(precompute),
            copy_X=params.boolean(copy_X),
            max_iter=params.integer(max_iter, from_=1),
            tol=params.real(tol, above=0.0),
            warm_start=params.boolean(warm_start),
            positive=params.boolean(positive),
            random_state=params.optional_(rng, params.integer),
            selection=params.enumeration(selection, {"cyclic", "random"}),
        )

    def fit(self, data: Data) -> "LassoSklearn":
        """Fits the model using training data.

        Parameters:
            data: tabular labeled data to train on

        Returns:
            self (allows chaining)
        """

        data = params.instance(data, Data)
        n = data.num_samples

        xtrain = params.real_matrix(data.samples(), nrows=n)
        ytrain = params.real_vector(data.labels(), dimensions=n)

        self._model.random_state = self.random.split(1)[0]
        self._model.fit(xtrain, ytrain)

        return self

    def apply(self, data: Data) -> DeltaPredictiveDistribution:
        r"""Predicts new inputs.

        Parameters:
            data: finite indexed data to predict;

        Returns:
            predictive normal distribution
        """

        data = params.instance(data, Data)
        xpred = params.real_matrix(data.samples())
        preds = self._model.predict(xpred)
        return DeltaPredictiveDistribution(mean=preds)
