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
    RandomVectorSampler
)


@dataclass
class OptimizerIteration:
    """Record the results of a single function evaluation during an optimization trajectory.

    Parameters:
        input: the input data, often a `TabularData` with one row
        output: the predicted distribution after applying the learner, often univariate
        score: the scalar-valued score of hthe output
    """

    input: Data
    output: PredictiveDistribution
    score: float


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
            Since optimizers minimize, if the goal is to maximize then we invert the score
    """

    def __init__(self, learner: Learner, scorer: Scorer, goal: str = "maximize"):
        self._learner = params.instance(learner, Learner)
        self._scorer = params.instance(scorer, Scorer)

        goal = params.enumeration(goal, {"maximize", "minimize"})
        if goal == "maximize":
            self._direction = -1
        elif goal == "minimize":
            self._direction = 1

        self._iterations = []

    def clear(self):
        """Reset the iterations to an empty list."""

        self._iterations = []

    @property
    def iterations(self) -> Sequence[OptimizerIteration]:
        """Sequence of what happened at each iteration of the optimizer."""

        return self._iterations

    def fit(self, data: Data):
        """Fit the learner on the data."""

        return self._learner.fit(data)

    def apply(self, data: Data) -> float:
        """Apply the learner and to produce an output distribution and score that distribution.
        Append the information about this iteration to the running list.
        Return a score that is modified so that lower is better, for use with optimizers.
        """

        dist = self._learner.apply(data)
        score = self._scorer.apply(dist)
        self._iterations.append(OptimizerIteration(data, dist, score))
        return score * self._direction


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


class RandomOptimizer(Optimizer):
    """Draws a random sample at each iteration.

    Parameters:
        num_iters: the number of random samples to draw
        domain: optional domain from which to draw values. If not provided, then the vector space
            dataset determines its own domain.
        rng: pseudo-random number generator
    """

    def __init__(self, num_iters: int, domain: Optional[Any] = None, rng=None, **kwargs):
        super.__init__(**kwargs)
        self._num_iters = params.integer(num_iters, above=0)
        self._sampler = RandomVectorSampler(size=1, domain=domain, rng=rng)

    def _optimize(self, data: VectorSpaceData, function_tracker: TrackedTransformation):
        """Generate random samples and evaluate each one.
        This ensures that each iteration is stored as an optimization iteration object.
        """
        for _ in range(self._num_iters):
            sample = self._sampler.apply(data)
            function_tracker.apply(sample)
