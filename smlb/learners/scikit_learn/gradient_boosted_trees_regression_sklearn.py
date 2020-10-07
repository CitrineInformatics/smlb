"""Gradient Boosted Trees regression, scikit-learn implementation.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2019-2020 Matthias Rupp, Citrine Informatics.

Wrapper for Gradient Tree Boosting / Gradient Boosted Decision Trees, scikit-learn implementation.
"""

from typing import Any, Optional, Union

from sklearn.ensemble import GradientBoostingRegressor

from smlb import (
    BenchmarkError,
    Data,
    DeltaPredictiveDistribution,
    NormalPredictiveDistribution,
    params,
    SupervisedLearner,
)

# todo: hyperparameter optimization.


class GradientBoostedTreesRegressionSklearn(SupervisedLearner):
    """Gradient-boosted trees regression, scikit-learn implementation.

    Additive model where in each stage a regression tree is fit on the negative gradient of the loss function.

    Supports only numeric (vector) inputs and labels.

    Based on

    Jerome H. Friedman: Greedy function approximation: A gradient boosting machine,
    Annals of Statistics 29(5): 1189-1232, 2001. URL https://projecteuclid.org/euclid.aos/1013203451

    Jerome H. Friedman: Stochastic Gradient Boosting, Computational Statistics & Data Analysis 38(4): 367-378, 2002.
    DOI 10.1016/S0167-9473(01)00065-2  A preprint of this publication seems to have been published in 1999.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    """

    def __init__(
        self,
        uncertainties: Optional[str] = None,
        loss: str = "ls",
        alpha: float = 0.9,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        n_estimators: int = 100,
        criterion: str = "mse",
        max_depth: int = 3,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, str, None] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        # min_impurity_split deprecated
        random_state: int = None,
        ccp_alpha: float = 0.0,
        init: Optional[Any] = None,
        validation_fraction: float = 0.1,
        n_iter_no_change: Optional[int] = None,
        tol: float = 0.0001,
        **kwargs,
    ):
        """Initialize state.

        sklearn-specific parameters are passed through to the implementation.

        Parameters:
            uncertainties: whether and how to compute predictive uncertainties; possible choices are
                None; by default, RandomForestRegressor does not return any predictive uncertainties;
            loss: loss function to optimize; valid values are "ls" (least squares), "lad" (least absolute deviation),
                "huber" (Huber's loss), "quantile" (quantile regression). Use alpha parameter for huber and quantile.
            alpha: quantile for "huber" and "quantile" loss functions
            learning_rate: value by which to shrink contribution of consecutive trees; trade-off with num_estimators
            subsample: fraction of samples for fitting base learners; if <1 results in Stochastic Gradient Boosting.
                reducing subsample reduces variance and increases bias.
            n_estimators: number of decision trees
            criterion: either Friedman improved score ("friedman_rmse"), variance reduction ("mse", mean squared error),
                or, mean absolute error ("mae")
            max_depth: maximum depth of a tree; default is 3
            min_samples_split: minimum number of samples required to split an internal node;
                float numbers indicate a fraction of number of training samples
            min_samples_leaf: minimum number of training samples required in a leaf node
                float numbers indicate a fraction of number of training samples
            min_weight_fraction_leaf: minimum weighted fraction of weights required in a leaf node
            max_features: number of features considered when splitting; integers directly specify the number,
                floating point values specify which fraction of all features to use;
                "auto" uses all features, "sqrt" and "log2" use square root and binary logarithm of number of features
            max_leaf_nodes: maximum number of leaves a tree can have
            min_impurity_decrease: minimum impurity decrease required for splitting
            random_state: pseudo-random number generator seed
            ccp_alpha: complexity parameter for minimal cost-complexity pruning.
            init: estimator for initial predictions; can be 'zero' for constant zero predictions
            validation_fraction: fraction of training data to set aside for early stopping; only with n_iter_no_change
            n_iter_no_change: set to integer to stop after no improvement (beyond tol) for that many rounds
            tol: tolerance for early stopping; only improvements larger than tol are considered

        The sklearn.GradientBoostingRegressor parameters `oob_score`, `verbose`, `warm_start` are not considered.

        See skl.ensemble.ExtraTreesRegressor parameters.
        """

        super().__init__(**kwargs)

        # validate parameters

        self._uncertainties = params.enumeration(uncertainties, {None})

        loss = params.enumeration(loss, {"ls", "lad", "huber", "quantile"})
        alpha = params.real(alpha, above=0, below=1)
        learning_rate = params.real(learning_rate, above=0, to=1)
        subsample = params.real(subsample, above=0, to=1)
        n_estimators = params.integer(n_estimators, from_=1)
        criterion = params.enumeration(criterion, {"friedman_rmse", "mse", "mae"})
        max_depth = params.any_(max_depth, lambda arg: params.integer(arg, from_=1), params.none)
        min_samples_split = params.any_(
            min_samples_split,
            lambda arg: params.integer(arg, from_=2),
            lambda arg: params.real(arg, above=0.0, to=1.0),
        )
        min_samples_leaf = params.any_(
            min_samples_leaf,
            lambda arg: params.integer(arg, from_=1),
            lambda arg: params.real(arg, above=0.0, to=1.0),
        )
        min_weight_fraction_leaf = params.real(min_weight_fraction_leaf, from_=0.0, to=1.0)
        max_features = params.any_(
            max_features,
            lambda arg: params.integer(arg, above=0),
            lambda arg: params.real(arg, above=0.0, to=1.0),
            lambda arg: params.enumeration(arg, {"auto", "sqrt", "log2"}),
            params.none,
        )
        max_leaf_nodes = params.any_(
            max_leaf_nodes, lambda arg: params.integer(arg, from_=1), params.none
        )
        min_impurity_decrease = params.real(min_impurity_decrease, from_=0.0)
        random_state = params.integer(random_state)
        ccp_alpha = params.real(ccp_alpha, from_=0.0)
        # no validation for init (no class signature validator)
        validation_fraction = params.real(validation_fraction, above=0, below=1)
        n_iter_no_change = params.any_(
            n_iter_no_change, lambda arg: params.integer(arg, from_=0), params.none
        )
        tol = params.real(tol, from_=0)

        self._model = GradientBoostingRegressor(
            loss=loss,
            alpha=alpha,
            learning_rate=learning_rate,
            subsample=subsample,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
            init=init,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
        )

    def fit(self, data: Data) -> "GradientBoostedTreesRegressionSklearn":
        """Fits the model using training data.

        Parameters:
            data: tabular labeled data to train on

        Returns:
            self (allows chaining)
        """

        data = params.instance(
            data, Data
        )  # todo: params.data(..., is_finite=True, is_labeled=True)
        n = data.num_samples

        xtrain = params.real_matrix(data.samples(), nrows=n)
        ytrain = params.real_vector(data.labels(), dimensions=n)

        self._model.fit(xtrain, ytrain)

        return self

    def apply(
        self, data: Data
    ) -> Union[DeltaPredictiveDistribution, NormalPredictiveDistribution]:
        r"""Predicts new inputs.

        Parameters:
            data: finite indexed data to predict;

        Returns:
            predictive normal distribution
        """

        data = params.instance(
            data, Data
        )  # todo: params.data(..., is_finite=True, is_labeled=True)

        xpred = params.real_matrix(data.samples())

        # predict
        # scikit-learn's ExtraTreesRegressor.predict() method does not support
        # returning predictions for all trees in the ensemble. Therefore,
        # `preds = self._model.predict(xpred)` is insufficient.

        if self._uncertainties is None:
            preds = self._model.predict(xpred)
            return DeltaPredictiveDistribution(mean=preds)
        elif self._uncertainties == "naive":
            # todo: there is a discrepancy between the ensemble mean and predictions
            #       until this has been resolved, naive uncertainties are not supported
            #       when fixing this, update parameter validation and unit tests
            raise NotImplementedError
        #     # #trees x #samples matrix of predictions of ensemble's trees
        #     staged_preds = np.asfarray(tuple(self._model.staged_predict(xpred)))

        #     # this does NOT yield the same predictions as self._model.predict(xpred)
        #     mean, stddev = (
        #         np.mean(staged_preds, axis=0),
        #         np.std(staged_preds, axis=0),
        #     )
        #     return NormalPredictiveDistribution(mean=mean, stddev=stddev)
        else:
            raise BenchmarkError(
                "internal error, unknown parameter for uncertainties of ExtremelyRandomizedTreesRegressionSklearn"
            )
