"""Tests of Optimizer implementations and the tracking of function evaluations.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
(c) 2019-20, Citrine Informatics.
"""

import numpy as np

from smlb import (
    TrackedTransformation,
    Learner,
    Data,
    NormalPredictiveDistribution,
    InvalidParameterError,
    ProbabilityOfImprovement,
    RandomOptimizer
)
from smlb.datasets.synthetic.friedman_1979.friedman_1979 import Friedman1979Data


class FakeLearner(Learner):
    """A fake implementation of the Learner class that predicts random values."""

    def apply(self, data: Data):
        if not data.is_finite:
            raise InvalidParameterError("a finite dataset", f"an infinite dataset of type {data.__class__}")

        means = np.random.uniform(0, 10, data.num_samples)
        stddevs = np.random.uniform(0.5, 2.0, data.num_samples)
        return NormalPredictiveDistribution(means, stddevs)


def test_tracking():
    """Test that every application of the learner is tracked when applying an optimizer."""
    num_samples = 5
    dataset = Friedman1979Data()
    learner = FakeLearner()
    scorer = ProbabilityOfImprovement(target=8.0, goal="maximize")

    func = TrackedTransformation(learner, scorer)
    optimizer = RandomOptimizer(num_samples=num_samples, rng=0)

    results1 = optimizer.optimize(data=dataset, function_tracker=func)
    # Re-applying the optimizer should clear the old results and store new ones.
    results2 = optimizer.optimize(data=dataset, function_tracker=func)

    for result in [results1, results2]:
        assert len(result.steps) == 1
        assert len(result.steps[0].scores) == num_samples
        assert result.num_evaluations == num_samples
    assert results1.steps[0].scores[0] != results2.steps[0].scores[0]
