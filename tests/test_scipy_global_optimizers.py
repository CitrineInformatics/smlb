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
from smlb.datasets.synthetic.friedman_1979.friedman_1979 import Friedman1979Data
from smlb.optimizers.scipy.global_optimizers import (
    ScipyDifferentialEvolutionOptimizer,
    ScipyDualAnnealingOptimizer
)


class IdentityLearner(Learner):
    """A fake implementation of the Learner class that predicts random values."""

    def __init__(self, function: VectorSpaceData, **kwargs):
        super().__init__(**kwargs)
        self._function = params.instance(function, VectorSpaceData)

    def apply(self, data: Data):
        if not data.is_finite:
            raise InvalidParameterError("a finite dataset", f"an infinite dataset of type {data.__class__}")
        means = self._function.labels(data.samples())
        stddevs = np.zeros_like(means)
        return NormalPredictiveDistribution(means, stddevs)


def test_optimizers():
    """Test something."""
    dataset = Friedman1979Data()
    learner = IdentityLearner(dataset)
    scorer = ExpectedValue()

    func = TrackedTransformation(learner, scorer, maximize=False)
    optimizer = ScipyDualAnnealingOptimizer(rng=0)

    results = optimizer.optimize(data=dataset, function_tracker=func)

    assert results.steps[0].scores[0]
