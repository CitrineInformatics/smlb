"""Optimization algorithms.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
Citrine Informatics 2019-2020
"""

from abc import ABCMeta, abstractmethod
from typing import Sequence, Optional, Any

import numpy as np

from smlb import (
    params,
    SmlbObject,
    Learner,
    Scorer,
    Data,
    TabularData,
    DataTransformation,
    PredictiveDistribution,
    VectorSpaceData,
    RandomVectorSampler,
    Random,
)


class OptimizerStep(SmlbObject):
    """Record the results of a set of function evaluations during an optimization trajectory.

    Parameters:
        input: the input data
        output: the predicted distribution after applying the learner
        scores: the scalar-valued scores of the output. This must be an array of length 1
            (implying the score is calculated on the entire batch of inputs) or of length
            equal to the number of samples in `data` (implying one score for each sample).
    """

    def __init__(
        self,
        input_: TabularData,
        output: PredictiveDistribution,
        scores: Sequence[float],
        **kwargs
    ):
        super().__init__(**kwargs)
        self._input: TabularData = params.instance(input_, TabularData)
        self._output: PredictiveDistribution = params.instance(output, PredictiveDistribution)
        # total number of function evaluations during this step
        self._num_evaluations: int = params.integer(self._input.num_samples, from_=1)
        self._scores: Sequence[float] = params.any_(
            scores,
            lambda arg: params.sequence(arg, length=1, type_=float),
            lambda arg: params.sequence(arg, length=self._num_evaluations, type_=float),
        )

    @property
    def num_evaluations(self) -> int:
        return self._num_evaluations

    @property
    def scores(self) -> Sequence[float]:
        return self._scores

    def scores_per_evaluation(self) -> Sequence[float]:
        """Return a sequence scores, with each score matched in order to the sequence of function evaluations.
        If there is one score, s, for the entire batch, return [s, s, s, ...]
        (one for each function evaluation).
        """
        if len(self._scores) == 1:
            return [self._scores[0]] * self.num_evaluations
        else:
            return self._scores


class OptimizerTrajectory:
    """Holds a single optimization trajectory, a sequence of OptimizerIterations."""

    def __init__(self, steps: Sequence[OptimizerStep]):
        self._steps = params.sequence(steps, type_=OptimizerStep)
        self._num_evaluations = np.sum([r.num_evaluations for r in self.steps])

    @property
    def steps(self) -> Sequence[OptimizerStep]:
        return self._steps

    @property
    def num_evaluations(self):
        """The total number of function evaluations across all of the steps."""
        return self._num_evaluations

    def best_score_trajectory(
        self, maximize: bool = True, length: Optional[int] = None
    ) -> Sequence[float]:
        """Calculate the best score found so far as a function of number of function evaluations.

        Parameters:
            maximize: whether the goal is to maximize (true) or minimize (false) the score
            length: total length of the result. If larger than the actual number of function
                evaluations, the result will be padded with the best value. If smaller than the
                actual number of evaluations, the result will be truncated.
                If None, the result is returned as-is.

        Returns:
            A sequence of floats, each one corresponding to the best score found at that point
            in the optimization trajectory.
        """
        maximize = params.boolean(maximize)
        length = params.optional_(length, lambda arg: params.integer(arg, from_=1))

        best_score = np.empty(self.num_evaluations)
        idx = 0
        best_score_so_far = self.steps[0].scores[0]
        direction = 1.0 if maximize else -1.0

        for optimization_iter in self.steps:
            for eval_ in optimization_iter.scores:
                if eval_ * direction > best_score_so_far * direction:
                    best_score_so_far = eval_
                best_score[idx] = best_score_so_far * direction
                idx += 1

        if length is not None:
            extra_padding = length - len(best_score)
            if extra_padding < 0:
                return best_score[:extra_padding]  # TODO: Raise a warning?
            return np.pad(best_score, ((0, extra_padding),), mode="edge")
        else:
            return best_score


class TrackedTransformation(DataTransformation):
    """A wrapper class that combines a `Learner` and a `Scorer`, recording the results
    and calculating the score every time the learner is applied. The score is adjusted
    such that *lower is better*.

    This is necessary to interface with optimizers such as those implemented in scipy,
    which require a scalar-valued function and also don't expose every step of the
    optimization trajectory.

    Parameters:
        learner: a Learner, to evaluate a specific sampled point
        scorer: a Scorer, to calculate a scalar-valued score at the point.
            This value is what the optimizer attempts to optimize.
        maximize: whether to maximize or minimize the score.
    """

    def __init__(self, learner: Learner, scorer: Scorer, maximize: bool = True):
        self._learner = params.instance(learner, Learner)
        self._scorer = params.instance(scorer, Scorer)

        maximize = params.boolean(maximize)
        # If the goal is to maximize the score, invert the value because optimizers minimize.
        if maximize:
            self._direction = -1
        else:
            self._direction = 1

        self._steps = []

    def clear(self):
        """Reset the steps to an empty list."""

        self._steps = []

    @property
    def steps(self) -> Sequence[OptimizerStep]:
        """Sequence of what happened at each step of the optimizer."""

        return self._steps

    @property
    def direction(self) -> float:
        """Whether it is better to maximize or minimize this score."""

        return self._direction

    def fit(self, data: Data):
        """Fit the learner on the data."""

        return self._learner.fit(data)

    def apply(self, data: Data) -> float:
        """Apply the learner to produce an output distribution and score that distribution.
        Append the information about this step to the running list.
        Return a score such that lower is always better.
        """

        dist = self._learner.apply(data)
        scores = self._scorer.apply(dist)
        self._steps.append(OptimizerStep(data, dist, scores))
        return scores * self._direction


class Optimizer(SmlbObject, metaclass=ABCMeta):
    """A scalar-valued optimizer that searches for the minimum value."""

    def optimize(
        self, data: VectorSpaceData, function_tracker: TrackedTransformation
    ) -> OptimizerTrajectory:
        """
        Run the optimization. This first clears the `function_tracker`'s memory of
        previous steps, then calls `_minimize`. The score must be configured such that
        lower scores are better.

        Parameters:
            data: vector space from which the optimizer can sample data
            function_tracker: an instance of TrackedTransformation to track function evaluations
                and scores. and a scorer, which converts the labeled data into a
                univariate score to minimize

        Returns:
            A sequence of OptimizerStep objects, one for every time the learner is applied
            to a dataset.
        """
        function_tracker.clear()
        self._minimize(data, function_tracker)
        return OptimizerTrajectory(function_tracker.steps)

    @abstractmethod
    def _minimize(self, data: VectorSpaceData, function_tracker: TrackedTransformation):
        """Perform the minimization."""

        raise NotImplementedError


class RandomOptimizer(Optimizer, Random):
    """Draws random samples.

    Parameters:
        num_samples: the number of random samples to draw
        domain: optional domain from which to draw values. If not provided, then the
            optimization domain is taken to be that of `data` parameter passed to `optimize()`.
        rng: pseudo-random number generator
    """

    def __init__(self, num_samples: int, domain: Optional[Any] = None, rng=None, **kwargs):
        super().__init__(rng=rng, **kwargs)
        self._num_samples = params.integer(num_samples, above=0)
        self._sampler = RandomVectorSampler(size=self._num_samples, domain=domain, rng=rng)

    def _minimize(self, data: VectorSpaceData, function_tracker: TrackedTransformation):
        """Generate num_samples random samples and evaluate them."""
        samples = self._sampler.apply(data)
        function_tracker.apply(samples)
