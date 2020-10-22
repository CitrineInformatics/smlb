"""Scorers, or acquisition functions in the context of Bayesian optimization.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
Citrine Informatics 2019-2020
"""

from abc import ABCMeta, abstractmethod
from typing import Sequence
import math

from scipy.special import erf
import numpy as np

from smlb import SmlbObject
from smlb import params
from smlb import PredictiveDistribution


class Scorer(SmlbObject, metaclass=ABCMeta):
    """Abstract base class for scorers.

    A score acts on a predictive distribution and returns a sequence of float-valued scores.

    A score might produce one score for each element of the distribution, or it might produce
    a single score for the entire distribution, representing a batch score.
    """

    @abstractmethod
    def apply(self, dist: PredictiveDistribution) -> Sequence[float]:
        """Applies the acquisition function to a distribution to produce a score.

        Parameters:
            dist: a distribution, generally produced by applying a regression model

        Returns:
            a floating-point score
        """
        raise NotImplementedError


class ExpectedValue(Scorer):
    """The score is equal to the predicted value.

    Parameters:
        maximize: whether higher values or lower values are better.
    """

    def __init__(self, maximize: bool = True, **kwargs):
        super().__init__(**kwargs)

        maximize = params.boolean(maximize)
        if maximize:
            self._direction = 1
        else:
            self._direction = -1

    def apply(self, dist: PredictiveDistribution) -> Sequence[float]:
        return dist.mean * self._direction


class ProbabilityOfImprovement(Scorer):
    """Likelihood of improvement beyond a univariate target.

    Parameters:
        target: floating-point target value to exceed
        goal: whether the goal is to find a value above the target (maximize)
            or below the target (minimize).
    """

    def __init__(self, target: float, goal: str = "maximize", **kwargs):
        super().__init__(**kwargs)

        self._target = params.real(target)
        goal = params.enumeration(goal, {"maximize", "minimize"})
        if goal == "maximize":
            self._direction = 1
        elif goal == "minimize":
            self._direction = -1

    def apply(self, dist: PredictiveDistribution) -> Sequence[float]:
        """Calculate the likelihood of the given distribution improving on the target value.
        This currently only works for normal distributions. To extend to non-normal distributions,
        we should have the `PredictiveDistribution` class expose a `cdf()` method.

        Parameters:
            dist: a univariate predictive distribution

        Returns:
             The probability mass of the distribution that is above/below the target
                (depending on if the goal is to maximize or minimize)
        """
        mean = params.real_vector(dist.mean)
        stddev = params.real_vector(dist.stddev, dimensions=len(mean), domain=(0, np.inf))

        # If the goal is to minimize, negate the target and the mean value.
        # Then, calculate the likelihood of improvement assuming maximization.
        target = self._target * self._direction
        mean = mean * self._direction
        return np.asfarray([self._calculate_li_above(m, s, target) for m, s in zip(mean, stddev)])

    @staticmethod
    def _calculate_li_above(mean, stddev, target):
        """Calculate the likelihood of improvement, assuming the goal is to exceed the target.

        Parameters:
            mean: mean of the normal distribution
            stddev: standard deviation of the normal distribution
            target: value to exceed
        """
        stddev = params.real(stddev, from_=0.0)
        if stddev == 0:
            if mean > target:
                return 1.0
            else:
                return 0.0
        return 0.5 * (1 - erf((target - mean) / (stddev * math.sqrt(2))))
