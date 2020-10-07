"""Extremely randomized trees, scikit-learn implementation.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2019-2020 Matthias Rupp, Citrine Informatics.

Extremely randomized trees regression as implemented by the scikit-learn ExtraTreesRegressor.
"""

from typing import Optional, Union

import numpy as np

from sklearn.ensemble import ExtraTreesRegressor

from smlb import (
    BenchmarkError,
    InvalidParameterError,
    params,
    SupervisedLearner,
    Data,
    DeltaPredictiveDistribution,
    NormalPredictiveDistribution,
    Random,
)

# todo: hyperparameter optimization.


class ExtremelyRandomizedTreesRegressionSklearn(SupervisedLearner, Random):
    """Extremely randomized trees regression, scikit-learn implementation.

    Ensemble of randomized decision trees via bootstrapping (sampling with replacement).

    Supports only numeric (vector) inputs and labels.

    Based on
    Pierre Geurts, Damien Ernst, Louis Wehenkel: Extremely randomized trees,
    Machine Learning 63: 3--42, 2006. DOI 10.1007/s10994-006-6226-1

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
    """

    def __init__(
        self,
        uncertainties: Optional[str] = None,
        n_estimators: int = 100,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, str, None] = "auto",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        # min_impurity_split deprecated
        bootstrap: bool = True,
        n_jobs: Optional[int] = None,
        random_state: int = None,
        ccp_alpha: float = 0.0,
        max_samples: Optional[Union[int, float]] = None,
        **kwargs,
    ):
        """Initialize state.

        sklearn-specific parameters are passed through to the implementation.
        The parameter `random_state` also acts as the seed `rng` for smlb's pseudo-random number
        generator and _must_ be specified.

        Parameters:
            uncertainties: whether and how to compute predictive uncertainties; choices are
                None; by default, RandomForestRegressor does not return predictive uncertainties;
                "naive"; uses the ensembles standard deviation
            n_estimators: number of decision trees
            criterion: either variance reduction ("mse", mean squared error), or, mean absolute error ("mae")
            max_depth: maximum depth of a tree; default is restricted only by min_samples_leaf
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
            bootstrap: if False, the whole dataset is used to build trees
            n_jobs: number of parallel jobs; -1 to use all available processors; None means 1
            random_state: pseudo-random number generator seed
            ccp_alpha: complexity parameter for minimal cost-complexity pruning.
            max_samples: number of input samples to draw during bootstrap; integers directly specify the number,
                floating point values specify which fraction of samples to use; all by default

        The sklearn.RandomForestRegressor parameters `oob_score`, `verbose`, `warm_restart` are not considered.

        See skl.ensemble.ExtraTreesRegressor parameters.
        """

        kwargs["rng"] = random_state
        super().__init__(**kwargs)

        # validate parameters

        self._uncertainties = params.enumeration(uncertainties, {None, "naive"})

        n_estimators = params.integer(n_estimators, from_=1)
        criterion = params.enumeration(criterion, {"mse", "mae"})
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
        bootstrap = params.boolean(bootstrap)
        n_jobs = params.any_(
            n_jobs,
            lambda arg: params.integer(arg, from_=-1, to=-1),
            lambda arg: params.integer(arg, from_=1),
            params.none,
        )
        random_state = params.integer(random_state)
        ccp_alpha = params.real(ccp_alpha, from_=0.0)
        max_samples = params.any_(
            max_samples,
            lambda arg: params.integer(arg, from_=1),
            lambda arg: params.real(arg, from_=0.0, to=1.0),
            params.none,
        )

        self._model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

    def fit(self, data: Data) -> "ExtremelyRandomizedTreesRegressionSklearn":
        """Fits the model using training data.

        Parameters:
            data: tabular labeled data to train on

        Returns:
            self (allows chaining)
        """

        data = params.instance(data, Data)

        if not data.is_labeled:
            raise InvalidParameterError("labeled data", "unlabeled data")
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

        data = params.instance(data, Data)

        xpred = params.real_matrix(data.samples())

        # predict
        # scikit-learn's ExtraTreesRegressor.predict() method does not support
        # returning predictions for all trees in the ensemble. Therefore,
        # `preds = self._model.predict(xpred)` is insufficient.

        if self._uncertainties is None:
            preds = self._model.predict(xpred)
            return DeltaPredictiveDistribution(mean=preds)
        elif self._uncertainties == "naive":
            preds = np.asfarray([tree.predict(xpred) for tree in self._model.estimators_])
            return NormalPredictiveDistribution(
                mean=np.mean(preds, axis=0), stddev=np.std(preds, axis=0)
            )
        else:
            raise BenchmarkError(
                "internal error, unknown parameter for uncertainties of ExtremelyRandomizedTreesRegressionSklearn"
            )
