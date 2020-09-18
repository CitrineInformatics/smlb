"""Scorers, or acquisition functions in the context of Bayesian optimization.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
Citrine Informatics 2019-2020
"""

from abc import ABCMeta, abstractmethod
import math

from scipy.special import erf

from smlb import (
    SmlbObject,
    params,
    PredictiveDistribution,
    BenchmarkError
)


class Scorer(SmlbObject, metaclass=ABCMeta):
    """Abstract base class for scorers.

    Many `Scorer`s act exclusively on univariate distributions and will raise a BenchmarkError
    if provided with a multivariate distribution as the input to their `apply` method.

    The `Scorer` concept can be extended to implementation that act on multivariate distributions,
    representing the score of a batch of candidates.
    """

    @abstractmethod
    def apply(self, dist: PredictiveDistribution) -> float:
        """Applies the acquisition function to a distribution to produce a score.

        Parameters:
            dist: a distribution, generally produced by applying a regression model

        Returns:
            a floating-point score
        """
        raise NotImplementedError


class LIScorer(Scorer):
    """Likelihood of improvement beyond a univariate target.

    Parameters:
        target: floating-point target value to exceed
        goal: whether the goal is to find a value above the target (maximize)
            or below the target (minimize). Tune objectives are not supported at this time.
    """

    def __init__(self, target: float, goal: str = "maximize", **kwargs):
        super().__init__(**kwargs)

        self._target = params.real(target)
        goal = params.enumeration(goal, {"maximize", "minimize"})
        if goal == "maximize":
            self._direction = 1
        elif goal == "minimize":
            self._direction = -1

    def apply(self, dist: PredictiveDistribution) -> float:
        """Calculate the likelihood of the given distribution improving on the target value.
        This currently only works for normal distributions. To extend to non-normal distributions,
        we should have the `PredictiveDistribution` class expose a `cdf()` method.

        Parameters:
            dist: a univariate predictive distribution

        Returns:
             The probability mass of the distribution that is above/below the target
                (depending on if the goal is to maximize or minimize)
        """
        mean = params.real_vector(dist.mean, dimensions=1)
        stdev = params.real_vector(dist.stddev, dimensions=1)

        if len(mean) != 1 or len(stdev) != 1:
            raise BenchmarkError(
                f"LI Scorer can only be applied to a univariate distribution. Got a distribution"
                f"of type {dist.__class__} with means {mean} and standard deviations {stdev}."
            )

        # If the goal is to minimize, negate the target and the mean value.
        # Then, calculate the integral from the target value to infinity for a normal distribution
        # with mean `mean` and standard deviation `stdev`.
        target = self._target * self._direction
        mean = mean[0] * self._direction
        stdev = params.real(stdev[0], from_=0)

        if stdev == 0:
            if mean > target:
                return 1.0
            else:
                return 0.0

        return 0.5 * (1 - erf((target - mean) / (stdev * math.sqrt(2))))
