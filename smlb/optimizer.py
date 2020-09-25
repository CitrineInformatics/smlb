"""Optimization algorithms.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
Citrine Informatics 2019-2020
"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Optional, Any

from smlb import (
    params,
    SmlbObject,
    Learner,
    Scorer,
    Data,
    DataTransformation,
    PredictiveDistribution,
    VectorSpaceData,
    RandomVectorSampler,
    Random
)


@dataclass
class OptimizerIteration:
    """Record the results of a set of function evaluations during an optimization trajectory.

    Parameters:
        input: the input data, often a `TabularData`
        output: the predicted distribution after applying the learner
        scores: the scalar-valued scores of the output
    """

    input: Data
    output: PredictiveDistribution
    scores: Sequence[float]


class TrackedTransformation(DataTransformation):
    """A wrapper class that combines a `Learner` and a `Scorer`, recording the results
    and calculating the score every time the learner is applied.

    This is necessary to interface with optimizers such as those implemented in scipy,
    which require a scalar-valued function and also don't expose every step of the
    optimization trajectory.

    Parameters:
        learner: a Learner, to evaluate a specific sampled point
        scorer: a Scorer, to calculate a scalar-valued score at the point.
            This value is what the optimizer attempts to optimize.
        goal: whether to "maximize" or "minimize" the score.
    """

    def __init__(self, learner: Learner, scorer: Scorer, goal: str = "maximize"):
        self._learner = params.instance(learner, Learner)
        self._scorer = params.instance(scorer, Scorer)

        goal = params.enumeration(goal, {"maximize", "minimize"})
        if goal == "maximize":
            self._direction = 1
        elif goal == "minimize":
            self._direction = -1

        self._iterations = []

    def clear(self):
        """Reset the iterations to an empty list."""

        self._iterations = []

    @property
    def iterations(self) -> Sequence[OptimizerIteration]:
        """Sequence of what happened at each iteration of the optimizer."""

        return self._iterations

    @property
    def direction(self) -> float:
        """Whether it is better to maximize or minimize this score."""

        return self._direction

    def fit(self, data: Data):
        """Fit the learner on the data."""

        return self._learner.fit(data)

    def apply(self, data: Data) -> float:
        """Apply the learner and to produce an output distribution and score that distribution.
        Append the information about this iteration to the running list.
        Return a score such that higher is always better.
        """

        dist = self._learner.apply(data)
        scores = self._scorer.apply(dist)
        self._iterations.append(OptimizerIteration(data, dist, scores))
        return scores * self._direction


class Optimizer(SmlbObject, metaclass=ABCMeta):
    """A scalar-valued optimizer that searches for the minimum value."""

    def optimize(
            self,
            data: VectorSpaceData,
            function_tracker: TrackedTransformation
    ) -> Sequence[OptimizerIteration]:
        """
        Run the optimization. This first clears the `function_tracker`'s
        memory of previous iterations, then calls `_optimize`.

        Parameters:
            data: vector space from which the optimizer can sample data
            function_tracker: a combination of a trained learner, which evaluates data points,
                and a scorer, which converts the labeled data into a univariate score to minimize

        Returns:
            A sequence of OptimizerIteration objects, one for every single function evaluation
                that was performed.
        """
        function_tracker.clear()
        self._optimize(data, function_tracker)
        return function_tracker.iterations

    @abstractmethod
    def _optimize(self, data: VectorSpaceData, function_tracker: TrackedTransformation):
        """Perform the optimization."""

        raise NotImplementedError


class RandomOptimizer(Optimizer, Random):
    """Draws a random sample at each iteration.

    Parameters:
        num_samples: the number of random samples to draw
        domain: optional domain from which to draw values. If not provided, then the
            dataset determines its own domain.
        rng: pseudo-random number generator
    """

    def __init__(self, num_samples: int, domain: Optional[Any] = None, rng=None, **kwargs):
        super().__init__(rng=rng, **kwargs)
        self._num_samples = params.integer(num_samples, above=0)
        self._sampler = RandomVectorSampler(size=self._num_samples, domain=domain, rng=rng)

    def _optimize(self, data: VectorSpaceData, function_tracker: TrackedTransformation):
        """Generate num_samples random samples and evaluate them."""
        samples = self._sampler.apply(data)
        function_tracker.apply(samples)
