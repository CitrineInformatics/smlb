"""Tests of Scipy's global optimizers

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
(c) 2019-20, Citrine Informatics.
"""

import pytest

import numpy as np

from smlb import (
    params,
    TrackedTransformation,
    VectorSpaceData,
    InvalidParameterError,
    ExpectedValue,
)
from smlb.optimizers.scipy.global_optimizers import (
    ScipyDifferentialEvolutionOptimizer,
    ScipyDualAnnealingOptimizer,
)
from smlb.learners.identity_learner import IdentityLearner


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
    """Test that the optimizers can be instantiated and run to find a global minimum."""
    dataset = DoubleWell()
    learner = IdentityLearner(dataset)
    scorer = ExpectedValue()
    func = TrackedTransformation(learner, scorer, maximize=False)

    optimizer_da = ScipyDualAnnealingOptimizer(rng=0, maxiter=10)
    optimizer_de = ScipyDifferentialEvolutionOptimizer(rng=0, maxiter=10)
    trajectory_da = optimizer_da.optimize(data=dataset, function_tracker=func)
    trajectory_de = optimizer_de.optimize(data=dataset, function_tracker=func)

    for trajectory in [trajectory_da, trajectory_de]:
        # find the lowest score and assert that it is within 1e-9 of the true min, 0.
        best_score = min([step.scores[0] for step in trajectory.steps])
        assert 0 <= best_score <= 1e-9

    # Multiple calls to the same optimizer should use different seeds.
    # Check that the two trajectories start from different points.
    trajectory_da_2 = optimizer_da.optimize(data=dataset, function_tracker=func)
    assert trajectory_da.steps[0].scores[0] != trajectory_da_2.steps[0].scores[0]


def test_optimizer_parameters():
    """For some of the more complicated parameters, test their validation."""
    # `rng` must be specified.
    with pytest.raises(InvalidParameterError):
        ScipyDualAnnealingOptimizer()
    with pytest.raises(InvalidParameterError):
        ScipyDifferentialEvolutionOptimizer()

    # `strategy` must be a string from the enumerated list.
    with pytest.raises(InvalidParameterError):
        ScipyDifferentialEvolutionOptimizer(rng=0, strategy="notastrategy")

    # `mutation` can either be a float or a 2-tuple. All numbers must be from [0, 2]
    # and in a tuple the first entry must not be greater than the second entry.
    valid_mutations = [0, 0.5, 2.0, (0, 1.0), (0, 2.0), (1.3, 1.3)]
    for mutation in valid_mutations:
        ScipyDifferentialEvolutionOptimizer(rng=0, mutation=mutation)

    invalid_mutations = [-3, 5.5, (-1, 1), (1, 3), (-1, 3), (1.5, 0.5)]
    for mutation in invalid_mutations:
        with pytest.raises(InvalidParameterError):
            ScipyDifferentialEvolutionOptimizer(rng=0, mutation=mutation)
