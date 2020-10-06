"""Random forest learner, 'lolo' implementation.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
2019-2020, Citrine Informatics.

Lolo is an implementation of random forest learners.
It is publicly available at https://github.com/CitrineInformatics/lolo
"""

from typing import Optional, Union

# requires lolopy
from lolopy.learners import BaseLoloLearner, RandomForestRegressor
from py4j.protocol import Py4JJavaError

from smlb import (
    BenchmarkError,
    Data,
    DeltaPredictiveDistribution,
    NormalPredictiveDistribution,
    params,
    PredictiveDistribution,
    SupervisedLearner,
)

# todo: hyperparameter optimization
# todo: handle randomness


class RandomForestRegressionLolo(SupervisedLearner):
    """Random forest regression, lolo implementation.

    See https://github.com/CitrineInformatics/lolo

    Supports only numeric (vector) inputs and labels.
    """

    def __init__(
        self,
        num_trees: int = -1,
        use_jackknife: bool = True,
        bias_learner: Optional[BaseLoloLearner] = None,
        leaf_learner: Optional[BaseLoloLearner] = None,
        subset_strategy: Union[str, int, float] = "auto",
        min_leaf_instances: int = 1,
        max_depth: int = 2 ** 30,
        uncertainty_calibration: bool = False,
        randomize_pivot_location: bool = False,
        # randomly_rotate_features: bool = False, currently in develop branch
        **kwargs
    ):
        """Initialize random forest model.

        See lolo Scala source code for initialization parameters:
        https://github.com/CitrineInformatics/lolo/blob/develop/src/main/scala/io/citrine/lolo/learners/RandomForest.scala

        When using `uncertainty_calibration=False` (the default), the number of trees
        `num_trees` should be set to a multiple of the number n of training samples,
        `num_trees = 4 * n` or higher. When using `uncertainty_calibration=True`,
        `num_trees = 64` is sufficient.

        Parameters:
            num_trees: number of trees in the forest; -1 uses number of training samples
            use_jackknife: whether to use jackknife-based variance estimates
            bias_learner: algorithm used to model bias
            leaf_learner: algorithm used at each leaf of the random forest
            subset_strategy: strategy to determine number of features used at each split
                "auto": use the default for lolo (all features for regression, sqrt for classification)
                "log2": use the base 2 log of the number of features
                "sqrt": use the square root of the number of features
                integer: set the number of features explicitly
                float: use a certain fraction of the features
            min_leaf_instances: minimum number of features used at each leaf
            max_depth: maximum depth of decision trees
            uncertainty_calibration: whether to empirically re-calibrate predicted uncertainties
                based on out-of-bag residuals
            randomize_pivot_location: whether to draw pivots randomly or always select the midpoint
            randomly_rotate_features: whether to rotate real scalar fetures for each tree
        """

        super().__init__(**kwargs)

        # validate parameters

        num_trees = params.any_(
            num_trees,
            lambda i: params.integer(i, above=0),
            lambda i: params.integer(i, from_=-1, to=-1),
        )

        use_jackknife = params.boolean(use_jackknife)

        bias_learner = params.any_(
            bias_learner, lambda arg: params.instance(arg, BaseLoloLearner), params.none
        )

        leaf_learner = params.any_(
            leaf_learner, lambda arg: params.instance(arg, BaseLoloLearner), params.none
        )

        subset_strategy = params.any_(
            subset_strategy,
            lambda s: params.enumeration(s, {"auto", "log2", "sqrt"}),
            lambda s: params.integer(s, above=0),
            lambda s: params.real(s, above=0),
        )

        min_leaf_instances = params.integer(min_leaf_instances, above=0)

        # the default 2**30 works for 32 bit or larger architectures
        max_depth = params.integer(max_depth, above=0)

        uncertainty_calibration = params.boolean(uncertainty_calibration)

        randomize_pivot_location = params.boolean(randomize_pivot_location)

        # randomly_rotate_features = params.boolean(randomly_rotate_features)

        # set up model

        try:
            self._model = RandomForestRegressor(
                num_trees=num_trees,
                use_jackknife=use_jackknife,
                bias_learner=bias_learner,
                leaf_learner=leaf_learner,
                subset_strategy=subset_strategy,
                min_leaf_instances=min_leaf_instances,
                max_depth=max_depth,
                uncertainty_calibration=uncertainty_calibration,
                randomize_pivot_location=randomize_pivot_location,
                # randomly_rotate_features=randomly_rotate_features,
            )
        except Py4JJavaError as e:
            raise BenchmarkError("instantiating lolo model failed") from e

        self._with_uncertainties = use_jackknife  # otherwise, deviations will be zero

    def fit(self, data: Data) -> "RandomForestRegressionLolo":
        """Fits the model using training data.

        Parameters:
            data: labeled tabular data to train on

        Returns:
            self (allows chaining)
        """

        data = params.instance(
            data, Data
        )  # todo: params.data(..., is_labeled=True, is_finite=True)
        n = data.num_samples

        xtrain = params.real_matrix(data.samples(), nrows=n)
        ytrain = params.real_vector(data.labels(), dimensions=n)

        try:
            self._model.fit(xtrain, ytrain)
        except Py4JJavaError as e:
            raise BenchmarkError("training lolo model failed") from e

        return self

    def apply(self, data: Data) -> PredictiveDistribution:
        """Predicts new inputs.

        Parameters:
            data: finite indexed data to predict

        Returns:
            predictive normal distributions if predictive uncertainties were requested,
            otherwise delta distributions
        """

        data = params.instance(
            data, Data
        )  # todo: params.data(..., is_labeled=True, is_finite=True)

        xpred = params.real_matrix(data.samples())

        if self._with_uncertainties:
            try:
                preds, stddevs = self._model.predict(xpred, return_std=True)
                return NormalPredictiveDistribution(mean=preds, stddev=stddevs)
            except Py4JJavaError as e:
                raise BenchmarkError("applying lolo model failed") from e
        else:
            try:
                preds = self._model.predict(xpred, return_std=False)
                return DeltaPredictiveDistribution(mean=preds)
            except Py4JJavaError as e:
                raise BenchmarkError("applying lolo model failed") from e
