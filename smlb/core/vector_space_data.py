"""Real vector space data.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp 2019-2020, Citrine Informatics.

See VectorSpaceData for details.
"""

from typing import Callable, Optional, Sequence, Tuple, TypeVar

import numpy as np

from smlb import BenchmarkError, InvalidParameterError
from smlb import Data
from smlb import TabularData
from smlb import params

I, L, S = TypeVar("I"), TypeVar("L"), TypeVar("S")


class VectorSpaceData(Data):
    """Real vector-space data, optionally labeled via dynamically evaluated functions.

    Data in a real finite-dimensional vector space.

    Labels can be computed dynamically by evaluating a function f.
    A hypercube domain can be specified for which f is valid.

    A vector space over a field F, here the real numbers, is a set closed under
    addition and multiplication with a scalar from F.

    This class is limited to real numbers and finite-dimensional spaces.
    For finite samples of vectors use `TabularData`.

    A hypercube domain can be specified to indicate valid sampling region.
    This is to enable restriction of samples to the domain of f.
    Bounded continuous subsets of vector spaces are not vector spaces.
    """

    def __init__(
        self,
        dimensions: int,
        function: Optional[Callable[[np.ndarray], Sequence[L]]] = None,
        domain: Optional[Sequence[Tuple[float, float]]] = None,
        **kwargs
    ):
        """Initialize vector space data.

        If no function is specified, data are unlabeled.
        If a domain is specified, samples must be within that domain.

        Parameters:
            dimensions: dimensionality of vector space; positive finite integer
            function: a function that accepts a real matrix (vectors are rows)
                and returns a corresponding sequence of labels.
                If not specified, Data are unlabeled.
            domain: domain in the form of a hypercube, if specified;
                given as a sequence of intervals [a,b], where a <= b.
                If only a single interval is specified it is used for all dimensions.

        Raises:
            InvalidParameterError for invalid arguments.
        """

        self._dimensions = params.integer(dimensions, above=0)
        self._function = params.optional_(
            function, lambda arg: params.callable(arg, num_pos_or_kw=1)
        )
        self._domain = params.optional_(
            domain, lambda arg: params.hypercube_domain(arg, self._dimensions)
        )

        super().__init__(*kwargs)

    @property
    def is_finite(self) -> bool:
        """Query for whether the data is finite or infinite.

        Returns:
            False
        """

        return False

    def samples(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Query vector samples.

        Returns a sequence of samples or raises InvalidParameterError.

        Vectors are queried by themselves, that is, vectors are their own indices.

        Parameter:
            indices: a real matrix of appropriate dimensions (rows are vectors)

        Return:
            real matrix (vectors are rows)

        Raises:
            InvalidParameterError: for invalid keys
        """

        samples = params.real_matrix(indices, ncols=self.dimensions)

        if self.domain is not None:
            if (samples < self._domain[:, 0]).any() or (samples > self._domain[:, 1]).any():
                raise InvalidParameterError("vectors in domain", "vectors outside of domain")

        return samples

    @property
    def num_samples(self) -> float:
        """Query number of samples in vector space.

        Returns:
            Infinity
        """

        return float("inf")

    def labels(self, indices: Optional[np.ndarray] = None) -> Sequence[L]:
        """Query computed labels.

        Returns a sequence of labels or raises InvalidParameterError.

        Parameters:
            indices: a real matrix of appropriate dimensions (rows are vectors)

        Returns:
            A sequence of labels

        Raises:
            InvalidParameterError: for invalid indices
            BenchmarkError: when querying labels for unlabeled data.
                If the label function returns too few or too many labels
            Any exception the label function raises.
        """

        if not self.is_labeled:
            raise BenchmarkError("querying labels for unlabeled data")

        inputs = self.samples(indices)
        labels = self._function(inputs)

        if len(labels) != len(indices):
            raise BenchmarkError("Label function returned wrong number of labels")

        return labels

    @property
    def is_labeled(self) -> bool:
        """Query for whether the data are labeled.

        Returns:
            True if data are labeled, False otherwise.
        """

        return self._function is not None

    def subset(
        self, indices: Optional[np.ndarray] = None, duplicates: bool = False
    ) -> TabularData:
        """Create finite subset of data.

        Parameters:
            indices: a real matrix of appropriate dimensions (rows are vectors)
            duplicates: if True (default), the returned subset does not contain
                duplicate entries; if False, duplicates are kept. Both inputs
                and labels have to match for duplicates.

        Returns:
            Finite dataset of vectors.
        """

        # indices is validated by calls to samples() and labels()
        duplicates = params.boolean(duplicates)

        data = self.samples(indices)
        labels = self.labels(indices) if self.is_labeled else None

        ds = TabularData(data=data, labels=labels)

        return ds if not duplicates else ds.subset(duplicates=True)

        # no specialized intersection method
        # no specialized complement method

    @property
    def dimensions(self) -> int:
        """Query number of dimensions.

        Returns:
            dimensionality of vector space
        """

        return self._dimensions

    @property
    def domain(self):
        """Query domain.

        Returns:
            vector space function domain; can be None
        """

        return self._domain  # can be None
