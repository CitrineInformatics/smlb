"""Sampling.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
2019-2020, Matthias Rupp, Citrine Informatics.

Sampling methods. Samplers are DataTransformations.

For some sampling methods, drawing the training set depends on the
validation set (or vice versa), for example, when randomly sampling
from a finite dataset as the intersection of training and validation
subsets must be empty. This is handled by the Workflow, not the Sampler:
After the validation set has been drawn, only the remaining samples
(set difference) is passed to the training set sampler.

Alternatives would have been (i) to couple or split two samplers;
this approach was discarded due to higher complexity. (ii) providing
a 'kind' argument to the initializer that indicates whether to sample
for 'validation' or 'training'. However, for randomized Samplers, the
exact same pseudo-random number seed would have to be passed, which
seems brittle and could lead to hard-to-find errors.

For cross-validated sampling, all relevant information, for example,
fold number in k-fold cross-validation, is specified at initialization.
"""

import itertools
from typing import Any, Optional

import numpy as np

from .exceptions import BenchmarkError, InvalidParameterError
from .parameters import params
from .random import Random
from .transformations import DataValuedTransformation
from .data import Data
from .vector_space_data import VectorSpaceData


class Sampler(DataValuedTransformation):
    """Abstract base class for all sampling transformations."""

    pass


class RandomSubsetSampler(Sampler, Random):
    """Draws a random subset from data.

    Applicable to TabularData, or any Data that accepts 0-based integer indices.
    """

    def __init__(self, size, **kwargs):
        """Initialize state.

        Parameters:
            size: number of samples to draw
            All arguments of Sampler and Random base classes.
        """

        super().__init__(**kwargs)

        self._size = params.integer(
            size, from_=0
        )  # partial validation (upper bound validated in apply)

    # no specialized fit() method as there is no internal state.

    def apply(self, data: Data, **kwargs) -> Data:
        """Draw random subset of data.

        Parameters:
            data: dataset to sample from

        Returns:
            random subset of data
        """

        data = params.instance(data, Data)
        if not data.is_finite:
            raise InvalidParameterError("finite Data", type(data).__name__)
        size = params.integer(
            self._size, from_=0, to=data.num_samples
        )  # validate upper bound (see __init__)

        ind = self.random.choice(data.num_samples, size=size, replace=False)

        return data.subset(ind)


class RandomVectorSampler(Sampler, Random):
    """Draws random vectors.

    Applicable to VectorSpaceData, or any Data that accepts vectors as indices.
    Draws random vectors from a hyper-rectangular region in a real vector space.
    """

    def __init__(self, size, domain: Optional[Any] = None, rng=None, **kwargs):
        """Initialize sampler.

        Parameters:
            size: number of vector samples to draw
            domain: (sub)domain to sample from; default is to use the data's domain
                if available, or the unit hypercube otherwise
            rng: pseudo-random number generator used

        Returns:
            IndexedFiniteData of vectors
        """

        super().__init__(rng=rng, **kwargs)

        self._size = params.integer(size, from_=0)  # no upper bound on number of vectors to draw
        self._domain = params.optional_(domain, lambda arg: params.hypercube_domain(arg))
        # partial validation; dimensionality is checked in apply() when data is known

    # no specialized fit() method as there is no internal state.

    def apply(self, data: Data, **kwargs) -> Data:
        """Draw random vectors.

        Parameters:
            data: Data to draw from

        Returns:
            TabularData of vectors
        """

        data = params.instance(data, Data)
        if self._domain is None:
            if data.domain is None:
                domain = np.asarray([[0, 1]] * data.dimensions)
            else:
                domain = data.domain
        else:
            domain = params.hypercube_domain(
                self._domain, dimensions=data.dimensions
            )  # checks dimensionality (see __init__)

        for low, high in domain:
            if low == -np.inf or high == np.inf:
                raise BenchmarkError("can not sample from infinite domain")

        # vectors = np.transpose(
        #     np.asfarray(
        #         [
        #             self.random.uniform(low=low, high=high, size=self._size)
        #             for (low, high) in self._domain
        #         ]
        #     )
        # )

        # this version avoids the python loop for efficiency in high dimensions
        vectors = (
            self.random.uniform(size=(self._size, data.dimensions)) * (domain[:, 1] - domain[:, 0])
            + domain[:, 0]  # noqa W503
        )

        return data.subset(vectors)


class GridSampler(Sampler, Random):
    """Evenly-spaced grid sampling in a real vector space.

    A specified number of samples are drawn from the smallest evenly-space grid of sufficient size.

    Applicable to VectorSpaceData.
    """

    def __init__(self, size, domain=None, rng=None, **kwargs):
        """Initialize state.

        Parameters:
            size: number of samples to draw
            domain: (sub)domain to sample from; by default dataset's domain is used
            rng: pseudo-random number generator seed
        """

        super().__init__(rng=rng, **kwargs)

        self._size = params.integer(size, from_=0)  # no upper bound for infinite spaces
        self._domain = domain

    def next_grid_size(self, data: VectorSpaceData, n: int):
        r"""Number of samples for smallest evenly-spaced grid with at least n vertices.

        \[ k = \ceil \sqrt[d](n) \rceil , \]

        where $d$ is the dimensionality of the vector space.

        Parameters:
            data: sampled dataset
            n: number of samples the grid must contain

        Returns:
            smallest number of samples per dimension for an evenly-spaced grid
            that has at least n points
        """

        n = params.integer(n, above=0)
        d = data.dimensions

        # fails for n = 3125 due to rounding error:
        # int(math.ceil(math.pow(n, 1./self.dimensions)))
        # works, but adds dependency:
        # k = decimal.Decimal(n) ** ( decimal.Decimal(1) / decimal.Decimal(self.dimensions) )
        # return int( k.to_integral_exact(rounding=decimal.ROUND_CEILING) )

        k = int(np.floor(np.power(float(n), 1.0 / d)))
        return k if k ** d >= n else k + 1

    def full_grid(self, data: VectorSpaceData, samples_per_dim: int, domain=None):
        """Full multi-dimensional evenly-spaced grid.

        For one sample per dimension, the result is a single vector, the mean of the domain.

        Parameters:
            data: sampled dataset
            samples_per_dim: number of evenly-spaced samples to take in each dimension
            domain: (sub)domain to sample from; by default, data's domain is used

        Returns:
            two-dimensional NumPy array where samples are rows
        """

        data = params.instance(data, VectorSpaceData)
        k = params.integer(samples_per_dim, above=0)  # positive integer
        domain = data.domain if domain is None else domain
        domain = params.hypercube_domain(domain, data.dimensions)

        if k == 1:
            return np.mean(domain, axis=1).reshape((1, -1))
        locs = (np.linspace(xfrom, xto, k) for xfrom, xto in domain)
        return np.asfarray(list(itertools.product(*locs)))

    # fit() is a no-op, use base class default

    def apply(self, data: VectorSpaceData):
        """Sample set from evenly-spaced grid in a vector space.

        A specified number of samples are drawn from the smallest
        evenly-space grid of sufficient size.

        Returns:
            sampled set

        If size does not correspond exactly to a k x ... x k grid, that
        is, if size is not a power of k, the next-largest grid of size
        k+1 x ... x k+1 is created and some of its samples are removed.
        Here, k denotes the number of evenly-spaced samples per dimension.
        """

        data = params.instance(data, VectorSpaceData)

        k = self.next_grid_size(data, self._size)
        population = self.full_grid(data, samples_per_dim=k, domain=self._domain)
        ind = self.random.choice(len(population), size=self._size, replace=False)

        return data.subset(population[ind])
