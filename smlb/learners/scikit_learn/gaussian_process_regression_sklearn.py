"""Gaussian Process Regression, scikit-learn implementation.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2019-2020, Citrine Informatics.

Gaussian process regression is a Bayesian kernel regression algorithm.
It is closely related to its Frequentist counterpart, Kernel Ridge Regression.
"""

from typing import Optional, Sequence, Union

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, RBF, WhiteKernel

from smlb import (
    Data,
    NormalPredictiveDistribution,
    params,
    SupervisedLearner,
)

# todo: hyperparameter optimization.
#       two modes should be supported:
#       sklearn-internal optimization, corresponding to a learner without HPs
#       smlb optimization, where HPs are not optimized internally by the sklearn GP
#       currently, only the first mode is supported

# todo: handle randomness via improved Random class


class GaussianProcessRegressionSklearn(SupervisedLearner):
    """Gaussian Process Regression, scikit-learn implementation.

    The default is a Gaussian kernel combined with a "White kernel" to model additive Gaussian noise.

    Supports only numeric (vector) inputs and labels.

    See
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.WhiteKernel.html#sklearn.gaussian_process.kernels.WhiteKernel
    """

    def __init__(
        self,
        internal_hp_optimization: bool = True,
        kernel: Optional[Kernel] = None,
        alpha: Union[float, Sequence] = 1e-5,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        random_state: int = None,
        **kwargs
    ):
        """Initialize state.

        sklearn-specific parameters are passed through to the implementation.

        Parameters:
            internal_hp_optimization: if True, hyperparameters are optimized "internally"
                by the Gaussian process, that is, scikit-learn optimizes hyperparameters
                and for smlb the learner has no hyperparameters;
                if False, hyperparameters are optimized by smlb (and scikit-learn does
                not optimize any hyperparameters)
            kernel: scikit-learn kernel; if None, a single Gaussian kernel is used as default
            alpha: regularization constant (scalar or vector); added as-is to kernel matrix diagonal.
                   Equivalent to adding a "WhiteKernel"; the default is the corresponding value from
                   scikit-learn's WhiteKernel, and different from scikit-learn's GaussianProcessRegressor.
            optimizer: hyperparameter optimization algorithm; used only if internal_hp_optimization is True
            n_restarts_optimizer: number of times optimizer is restarted; only used if internal_hp_optimization is True
            normalize_y: whether to subtract the mean of the labels
            random_state: integer seed

        See skl.gaussian_process.GaussianProcessRegressor parameters.
        """

        super().__init__(**kwargs)

        internal_hp_optimization = params.boolean(internal_hp_optimization)
        kernel = params.any_(kernel, lambda arg: params.instance(arg, Kernel), params.none)
        # incomplete check for alpha as dimension becomes known only at fitting time
        alpha = params.any_(
            alpha,
            lambda arg: params.real(arg, from_=0),
            lambda arg: params.real_vector(arg, domain=[0, np.inf]),
        )
        # todo: check optimizer, requires params.union (of string and callable) and params.function
        normalize_y = params.boolean(normalize_y)
        random_state = params.integer(random_state)

        if kernel is None:
            kernel = RBF() + WhiteKernel()

        assert internal_hp_optimization is True  # external HP optimization not yet supported

        self._model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            random_state=random_state,
        )

    def fit(self, data: Data) -> "GaussianProcessRegressionSklearn":
        """Fits the model using training data.

        Parameters:
            data: labeled data to train on;
                  must derive from IndexedData and LabeledData

        Returns:
            self (allows chaining)
        """

        data = params.instance(data, Data)  # todo: params.data(..., is_finite=True, is_labeled=True)
        n = data.num_samples

        xtrain = params.real_matrix(data.samples(), nrows=n)
        ytrain = params.real_vector(data.labels(), dimensions=n)

        self._model.fit(xtrain, ytrain)

        return self

    def apply(self, data: Data) -> NormalPredictiveDistribution:
        r"""Predicts new inputs.

        For Gaussian processes, both the noise-free predictive (posterior) 
        distribution as well as the noise estimate are normally distributed.
        The predictive distribution with noise is the sum of the former two.

        The $\alpha$ training noise specified at initialization time is not
        added at prediction time, and thus not part of the noise model.
        The current implementation considers contributions from any 
        WhiteKernel or other kernel that has a hyperparameter 'noise_level'.

        Limitations:
            It is a currently accepted shortcoming that WhiteKernels that are
            not 'first-level' sum members might yield wrong noise models.
            Examples: 
            WhiteKernel(...) + other kernels will work
            kernel(...) * WhiteKernel(...) will not work as intended

            Training data noise $\alpha$ is not added 

        Parameters:
            data: finite indexed data to predict;

        Returns:
            predictive normal distribution with the following decomposition:
                predicted: sum of model and noise distribution
                noise_part: normal distribution for estimated noise
                signal_part: normal distribution for estimated model contribution;
                             the Gaussian process' "predictive variance";
                             depends only on distance from the training data
        """

        data = params.instance(data, Data)  # todo: params.data(..., is_finite=True, is_labeled=True)

        xpred = params.real_matrix(data.samples())
        n = data.num_samples

        # predict
        preds, stddevs = self._model.predict(xpred, return_std=True)

        # noise
        # noise are all noise_level of WhiteKernel, where noise_level is variance (not standard deviation)
        # this assumes that the noise level are independent
        noise = tuple(
            v for k, v in self._model.kernel_.get_params().items() if k.endswith("noise_level")
        )
        noise = np.ones(shape=n) * np.sum(noise)
        noise_part = NormalPredictiveDistribution(mean=np.zeros(shape=n), stddev=np.sqrt(noise))

        return NormalPredictiveDistribution(
            mean=preds,
            stddev=np.sqrt(np.square(stddevs) + noise),
            noise_part=noise_part,
            signal_part=NormalPredictiveDistribution(mean=preds, stddev=stddevs),
        )
