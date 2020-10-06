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
    Random
)


class OptimizerIteration(SmlbObject):
    """Record the results of a set of function evaluations during an optimization trajectory.

    Parameters:
        input: the input data
        output: the predicted distribution after applying the learner
        scores: the scalar-valued scores of the output. This must be an array of length 1
            (implying the score is calculated on the entire batch of inputs) or of length
            equal to the number of samples in `data` (implying one score for each sample).
    """

    def __init__(self,
                 input_: TabularData,
                 output: PredictiveDistribution,
                 scores: Sequence[float],
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self._input: TabularData = params.instance(input_, TabularData)
        self._output: PredictiveDistribution = params.instance(output, PredictiveDistribution)
        # total number of function evaluations during this iteration
        self._num_evaluations: int = params.integer(self._input.num_samples, from_=1)
        self._scores: Sequence[float] = params.any_(
            scores,
            lambda arg: params.sequence(arg, length=1, type_=float),
            lambda arg: params.sequence(arg, length=self._num_evaluations, type_=float)
        )

    @property
    def num_evaluations(self) -> int:
        return self._num_evaluations

    @property
    def scores(self) -> Sequence[float]:
        return self._scores

    def scores_per_evaluation(self) -> Sequence[float]:
        """Return a sequence of scores that correspond one-to-one with evaluations.
        If there is one score, s, for the entire batch, return [s, s, s, ...]
        (one for each function evaluation).
        """
        if len(self._scores) == 1:
            return [self._scores[0]] * self.num_evaluations
        else:
            return self._scores


class OptimizerResults:
    """Holds the complete results of an optimization run, a sequence of OptimizerIterations."""

    def __init__(self, iterations: Sequence[OptimizerIteration]):
        self._iterations = params.sequence(iterations, type_=OptimizerIteration)
        self._num_evaluations = np.sum([r.num_evaluations for r in self.iterations])

    @property
    def iterations(self) -> Sequence[OptimizerIteration]:
        return self._iterations

    @property
    def num_evaluations(self):
        """The total number of function evaluations across all of the iterations."""
        return self._num_evaluations

    def best_score_trajectory(self,
                              maximize: bool = True,
                              length: Optional[int] = None
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
        best_score_so_far = self.iterations[0].scores[0]
        direction = 1.0 if maximize else -1.0

        for optimization_iter in self.iterations:
            for eval_ in optimization_iter.scores:
                if eval_ * direction > best_score_so_far * direction:
                    best_score_so_far = eval_
                best_score[idx] = best_score_so_far * direction
                idx += 1

        if length is not None:
            extra_padding = length - len(best_score)
            if extra_padding < 0:
                return best_score[:extra_padding]  # TODO: Raise a warning?
            return np.pad(best_score, ((0, extra_padding),), mode='edge')
        else:
            return best_score


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
        """Apply the learner to produce an output distribution and score that distribution.
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
    ) -> OptimizerResults:
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
        return OptimizerResults(function_tracker.iterations)

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
