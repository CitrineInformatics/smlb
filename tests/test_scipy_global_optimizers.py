"""Tests of Scipy's global optimizers

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
(c) 2019-20, Citrine Informatics.
"""

import numpy as np

from smlb import (
    params,
    TrackedTransformation,
    Learner,
    Data,
    VectorSpaceData,
    NormalPredictiveDistribution,
    InvalidParameterError,
    ExpectedValue,
)
from smlb.optimizers.scipy.global_optimizers import (
    ScipyDifferentialEvolutionOptimizer,
    ScipyDualAnnealingOptimizer,
)


# TODO: Implement this as a learner
class IdentityLearner(Learner):
    """A fake implementation of the Learner class that predicts random values."""

    def __init__(self, function: VectorSpaceData, **kwargs):
        super().__init__(**kwargs)
        self._function = params.instance(function, VectorSpaceData)

    def apply(self, data: Data):
        if not data.is_finite:
            raise InvalidParameterError(
                "a finite dataset", f"an infinite dataset of type {data.__class__}"
            )
        means = self._function.labels(data.samples())
        stddevs = np.zeros_like(means)
        return NormalPredictiveDistribution(means, stddevs)


class DoubleWell(VectorSpaceData):
    r"""Basic 2-dimensions function with 2 wells for testing global optimizers.

    \[ f(x, y) = x^4/4 + x^3/3 - 2x^2 - 4x + 28/3 + y^2 \]
    Limited to [-3, 3]^2.

    Global minimum is at (2, 0) with a value of 0.
    Global maximum is at (-3, +/-3) with a value of 283/12.
    """

    def __init__(self, **kwargs):
        """Initialize state."""

        dimensions = 2
        domain = params.hypercube_domain((-3, 3), dimensions=dimensions)

        super().__init__(
            dimensions=dimensions, function=self.__class__.doubleWell, domain=domain, **kwargs
        )

    @staticmethod
    def doubleWell(xx):
        """Computes double well test function without noise term.

        Parameters:
            xx: sequence of vectors

        Returns:
            sequence of computed labels
        """

        xx = params.real_matrix(xx)  # base class verifies dimensionality and domain

        x = xx[:, 0]
        y = xx[:, 1]
        return (
            1 / 4 * np.power(x, 4)
            + 1 / 3 * np.power(x, 3)
            - 2 * np.power(x, 2)
            - 4 * x
            + np.power(y, 2)
            + 28 / 3
        )


def test_optimizers_run():
    """Test that the optimizers can be instantiated and run without error."""
    dataset = DoubleWell()
    learner = IdentityLearner(dataset)
    scorer = ExpectedValue()

    func = TrackedTransformation(learner, scorer, maximize=False)
    optimizer1 = ScipyDualAnnealingOptimizer(rng=0, maxiter=10)
    optimizer2 = ScipyDifferentialEvolutionOptimizer(rng=0, maxiter=10)

    optimizer1.optimize(data=dataset, function_tracker=func)
    optimizer2.optimize(data=dataset, function_tracker=func)
