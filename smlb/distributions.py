"""Predictive distributions.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp 2019-2020, Citrine Informatics.

Predictive statistical distributions.
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from .object import SmlbObject
from .exceptions import BenchmarkError
from .parameters import params

# Notes:
# There is currently no division into continuous and discrete distributions.
# Nor are there currently classes for multi-variate distributions.
# Both can be inserted if and when required.


class PredictiveDistributionDecomposition(SmlbObject):
    """Base class for decomposition functionality of predictive distributions.

    The default behaviour provided by this base class is of a non-decomposable distribution.
    Learners that provide decompositions of their predictive distributions can provide these
    via initializer arguments.

    This interface defines the valid predictive distribution components in smlb:

    predicted    this is always the predictive distribution returned by the learner;
                 related to prediction intervals
    noise_part   estimated noise if available; in a loose sense, the aleatoric component
    signal_part  estimated signal if available; in a loose sense, the epistemic component;
                 related to confidence estimates
    """

    def __init__(self, noise_part=None, signal_part=None, **kwargs):
        """Initialize decompositions.

        Parameters:
            noise_part: estimated noise distribution; the aleatoric component
            signal_part: estimated signal distribution; the epistemic component
        """

        super().__init__(**kwargs)

        optional = lambda arg: params.any_(
            arg, lambda x: params.instance(x, PredictiveDistribution), params.none
        )
        self._noise_part = optional(noise_part)
        self._signal_part = optional(signal_part)

        pass

    @property
    def predicted(self):
        """Query predictive distribution."""

        return self

    @property
    def has_noise_part(self):
        """True if noise part of decomposition is available."""

        return self._noise_part is not None

    @property
    def noise_part(self):
        """Query noise part of decomposition.

        Raises:
            BenchmarkError: if distribution does not provide noise part
        """

        if self._noise_part is None:
            raise BenchmarkError("Distribution does not provide noise part decomposition")

        return self._noise_part

    @property
    def has_signal_part(self):
        """True if signal part of decomposition is available."""

        return self._signal_part is not None

    @property
    def signal_part(self):
        """Query signal part of decomposition.

        Raises:
            BenchmarkError: if distribution does not provide signal part
        """

        if self._signal_part is None:
            raise BenchmarkError("Distribution does not provide signal part decomposition")

        return self._signal_part


class PredictiveDistribution(PredictiveDistributionDecomposition, metaclass=ABCMeta):
    """Abstract base class for predictive distributions.
    
    A sequence of same-type distributions with possibly different parameters.
    For example, a sequence of normal distributions, each with its own mean and standard deviation.
    """

    @property
    @abstractmethod
    def mean(self):
        """Predictive means."""

        raise NotImplementedError

    @property
    @abstractmethod
    def stddev(self):
        """Predictive standard deviation."""

        raise NotImplementedError


class DeltaPredictiveDistribution(PredictiveDistribution):
    """(Dirac) delta predictive distributions.

    Use for deterministic values.

    A sequence of delta predictive distributions.

    In some (but not all) contexts it can make sense to view this distribution 
    as the limiting case of a normal distribution with standard deviation going to zero.
    """

    def __init__(self, mean, **kwargs):
        """Initialize state.
        
        Parameters:
            mean: sequence of means (floats)
        """

        super().__init__(**kwargs)

        self._mean = params.real_vector(mean)

    @property
    def mean(self):
        """Means of predictive distributions."""

        return self._mean

    @property
    def stddev(self):
        """Standard deviations of predictive distributions."""

        return np.zeros_like(self._mean)


# The normal predictive distributions are currently independent.
# This could be changed, for example, to accommodate the full posterior of a Gaussian process,
# that is, including the covariances between predictions.


class NormalPredictiveDistribution(PredictiveDistribution):
    """Normal predictive distributions.
    
    A sequence of (independent) normal predictive distributions.
    """

    def __init__(self, mean, stddev, **kwargs):
        """Initialize state.

        The normal distribution is completely characterized by
        its mean and standard deviation.

        Parameters:
            mean: a sequence of means (floats)
            stddev: a sequence of standard deviations (non-negative floats)
        """

        super().__init__(**kwargs)

        self._mean = params.real_vector(mean)
        self._stddev = params.real_vector(stddev, dimensions=len(self._mean), domain=(0, np.inf))

    @property
    def mean(self):
        """Means of predictive distributions."""

        return self._mean

    @property
    def stddev(self):
        """Standard deviations of predictive distributions."""

        return self._stddev
