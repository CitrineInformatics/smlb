"""Noise models.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
Citrine Informatics, 2019.

Provides:
* Noise models are specific types of randomness, for example, a normal distribution
* LabelNoise are data transformations that add Noise to labels of labeled data; 
  they are explicit steps in an experiment's workflow.
* InputNoise are data transformations that do the same for samples

LabelNoise and InputNoise models will add noise _every time_ samples or labels are
queried. For example, querying the label of a sample twice will return different values.
"""

from abc import ABCMeta, abstractmethod
import copy

import numpy as np

from smlb import BenchmarkError, InvalidParameterError
from smlb import params
from smlb import Data
from smlb import DataValuedTransformation
from smlb import Random


class Noise(Random, metaclass=ABCMeta):
    """Noise model.

    Abstract base class for all noise models.
    """

    @abstractmethod
    def noise(self, shape=None):
        """Generate noise.

        Parameters:
            shape: shape of noise vector, matrix or higher-order tensor

        Returns:
            a numerical array of given shape containing samples from a noise distribution

        Raises:
            InvalidParameterError: for invalid parameters
        """

        raise NotImplementedError


class NoNoise(Noise):
    """Noise-free.

    Returns empty (zero) array.

    This is a convenience class for situations where an explicit noise model is required.
    """

    def __init__(self, value: float = 0, **kwargs):
        """Initialize state.

        Parameters:
            value: constant that will be returned
            All parameters from base class 'Noise' initializer
        """

        super().__init__(**kwargs)

        self._value = params.real(value)

    def noise(self, shape=None):
        """Return no noise.

        A constant value is returned.

        Parameters:
            shape: shape of noise vector, matrix or higher-order tensor

        Returns:
            a numerical array of given shape containing a constant value

        Raises:
            InvalidParameterError: for invalid parameters
        """

        # valid shape are either positive integer or a tuple of positive integer
        is_nonneg_int = lambda arg: params.integer(arg, from_=1)
        is_tuple = lambda arg: params.tuple_(arg, is_nonneg_int)
        shape = params.any_(shape, is_nonneg_int, is_tuple)

        return np.full(shape, self._value)


class NormalNoise(Noise):
    """Homoskedastic, independent, normally distributed noise."""

    def __init__(self, mean=0.0, stddev=1.0, **kwargs):
        """Initialize state.

        Parameters:
            mean: mean of the normal distribution
            stddev: standard deviation of the normal distribution
            All parameters from base class 'Noise' initializer
        """

        super().__init__(**kwargs)

        self._mean = params.real(mean)
        self._stddev = params.real(stddev, above=0)

    def noise(self, shape=None):
        """Add Gaussian noise to labels.

        Parameters:
            shape: shape of noise vector, matrix or higher-order tensor

        Returns:
            a numerical array of given shape containing independent
            identically distributed Gaussian noise

        Raises:
            InvalidParameterError: for invalid parameters
        """

        # valid shape are either positive integer or a tuple of positive integer
        is_nonneg_int = lambda arg: params.integer(arg, from_=1)
        is_tuple = lambda arg: params.tuple_(arg, is_nonneg_int)
        shape = params.any_(shape, is_nonneg_int, is_tuple)

        return self.random.normal(self._mean, self._stddev, size=shape)


class LabelNoise(DataValuedTransformation):
    """Transform Data by adding Noise to labels."""

    def __init__(self, noise: Noise, **kwargs):
        """Initialize state.

        Parameters:
            noise: noise model

        Returns:
            dataset with noisy labels
        """

        super().__init__(**kwargs)

        self._noise = params.instance(noise, Noise)

    def apply(self, data: Data) -> Data:
        """Transforms data.

        Parameters:
            data: labeled data to transform

        Returns:
            transformed data

        Raises:
            InvalidParameterError if Data is not labeled
        """

        data = params.instance(data, Data)
        if not data.is_labeled:
            raise InvalidParameterError("labeled data", "unlabeled data")

        # patch the labels() method of the data object (not class)
        # there is no need to store the old labels function as it is a class member, not an object member

        for name in ("_orig_labels", "labels", "_noise"):
            # patch if necessary by choosing a random name instead of _labels
            if name in data.__dict__:
                raise BenchmarkError(f"internal error: data object already has {name} method")

        # create a copy of the dataset
        data = copy.deepcopy(data)

        # rename labels to _labels for data only
        setattr(data, "_orig_labels", getattr(data, "labels"))

        # store noise model
        setattr(data, "_noise", self._noise)

        # add wrapper as new labels() method

        def labels(self, indices=None):
            """Query labels of a sequence of samples.

            This wrapper adds noise.

            Parameters:
                indices: a sequence of sample 'indices'.
                         See 'samples()' for details.

            Returns:
                a sequence of labels
            """

            labels = self._orig_labels(indices)
            return labels + self._noise.noise(labels.shape)

        setattr(data, "labels", labels.__get__(data))

        return data
