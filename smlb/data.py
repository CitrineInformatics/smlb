"""Data.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp 2019-2020, Citrine Informatics.

Data sources and datasets. See class Data for details.
"""

# Supported:
# - Unlabeled and labeled data
# - Set operations
#
# Not supported:
# - Partially labeled data (semi-supervised learning)
#   One way to implement this would be to return None or a "missing" value
#   for unlabeled samples.
# - Multiset operations (proper treatment of duplicates)
#   Prepared, but specialized implementations still missing
# - (Multi)set operations involving different types of Data
#   For example, intersection of vector space and finite set of vector.
#   Prepared, but not implemented. Could be based, for example,
#   on lazy evaluation, using a Data type that holds references
#   to both lhs and rhs, and for each query checks whether it is
#   an element of both lhs and rhs.

from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, TypeVar, Union

from smlb.object import SmlbObject
from smlb.exceptions import InvalidParameterError
from smlb.parameters import params

# type annotation variables for indices (I), samples (S), and labels (L)
I, L, S = TypeVar("I"), TypeVar("L"), TypeVar("S")


class Data(SmlbObject, metaclass=ABCMeta):
    """Interface for data sources (datasets).

    The minimal interface supported by all data. It supports basic querying of
    samples, labels and dataset properties, as well as basic (multi)set operations.
    Derived classes can add additional methods.

    Set operations are required for value-based (as opposed to index-based)
    machine-learning workflows, for example, intersection to verify that a
    training and a validation set have no overlap. smlb is based on values,
    not indices, to emphasize correctness over efficiency.

    Set operations are provided as free functions (as opposed to class methods)
    to keep the interface lean and to reflect that set operations are not part
    of data but operate on data.

    In most settings, duplicate data points are not desirable, but in some
    settings they are. The Data class therefore does not enforce either:
    Specific Data classes and Workflows can support duplicates, or disallow them.
    Set operations ('duplicates=False') are provided for duplicate-free settings;
    multiset versions ('duplicates=True') are provided for settings with duplicates.

    For general datasets, set operations can be non-straightforward to implement,
    for example, intersection of infinite and finite data. smlb currently focuses
    on finite subsets, one reason being that finite subsets of infinite sets
    have measure zero.

    Any Data instance with zero samples (num_samples == 0) represents the empty set.

    Data classes that do not directly derive from this class should use the
    'register' method to register themselves as "virtual" subclasses.
    See Python documentation for the ABCMeta class' register method.
    """

    # note: comparing samples (or labels) is not part of the interface as
    #       it depends on context, even for the same dataset. As an example,
    #       whether two molecules with same stoichiometry but different bonding
    #       should be considered equal can depend on featurization. Consequently,
    #       sample (or label) comparison functions should be passed as
    #       arguments where required, and not be members of Data.

    # note: it is usually important whether a view or a copy of array dadta is returned.
    #       smlb follows an immutability policy, which makes this less relevant.

    @property
    @abstractmethod
    def is_finite(self) -> bool:
        """Query for whether the data is finite or infinite.

        Returns:
            True if data contains only finitely many samples,
            False if it contains infinitely many samples.

        There is currently no distinction between countably and uncountably infinite data.
        """

        raise NotImplementedError

    @abstractmethod
    def samples(self, indices: Optional[Sequence[I]] = None) -> Sequence[S]:
        """Query samples.

        Returns a sequence of samples or raises InvalidParameterError.

        'indices' are generalized indices; their type I depends on the data.
        For example, for indexed data, I could be a non-negative integer;
        for vector spaces, I could be a one-dimensional NumPy array of floats.
        For other datasets, I could be a string. Generalized indices must
        be unique for each dataset.

        Parameters:
            indices: A sequence of generalized 'indices' that specifies which
                samples to return. The type of `indices` depends on the data.
                By default, all samples are returned.

        Returns:
            Sequence of queried samples. Type of samples depends on the data.

        Raises:
            InvalidParameterError: for invalid keys
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def num_samples(self) -> Union[int, float]:
        """Query number of samples.

        Returns:
            Number of samples in data; guaranteed to be non-negative.
            Returns float('inf') for infinite data.
        """

        raise NotImplementedError

    @abstractmethod
    def labels(self, indices: Optional[Sequence[I]] = None) -> Sequence[L]:
        """Query labels.

        Returns a sequence of labels or raises InvalidParameterError.

        Parameters:
            indices: A sequence of generalized 'indices' that specifies which
                samples to return. See 'samples()' for details.

        Returns:
            A sequence of labels

        Raises:
            InvalidParameterError: for invalid keys, or keys of samples that
                do not have labels. Querying labels for unlabeled data
                always raises.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def is_labeled(self) -> bool:
        """Query for whether the data are labeled.

        Returns:
            True if data are labeled, False otherwise.
        """

        raise NotImplementedError

    @abstractmethod
    def subset(self, indices: Optional[Sequence[I]] = None, duplicates: bool = False) -> "Data":
        """Create finite subset of data.

        Parameters:
            indices: A sequence of generalized 'indices' that specifies which
                samples to include. See 'samples()' for details.
            duplicates: if False (default), the returned subset does not contain
                duplicate entries; if True, duplicates are kept. Both inputs
                and labels have to match for duplicates.

        Returns:
            Data containing only the specified samples, either without duplicates
            (subset) or taking duplicates into account (multiset subset).
            The type of the returned data can be the same as `type(self)`,
            but does not have to be. It should be conceptually as close to
            type of self  as possible. For example, if self is labeled, returned
            data should also be labeled. However, even if self is infinite, the
            returned type will often be finite.

        While there is technically no order in a set, subset() tries to maintain
        samples in the order of the indices where possible.
        """

        raise NotImplementedError

    @staticmethod
    def _intersection(lhs: "Data", rhs: "Data", duplicates: bool = False) -> "Data":
        """Specialized (multi)set intersection.

        This intersection method does not retain duplicates by default.
        For multiset behaviour, specify 'duplicates=True'.

        Derived classes can, but do not have to, override this method to
        provide an efficient specialized set intersection method. Such
        methods should validate the type of both arguments, and, if they
        can not compute the set intersection for them, raise NotImplementedError.

        Parameters:
            lhs: one of the two datasets to intersect ('left hand side')
            rhs: one of the two datasets to intersect ('right hand side')
            duplicates: if False (default), the returned data do not contain
                duplicate entries; if True, duplicates are taken into account.
                Both inputs and labels have to match for duplicates.

        Returns:
            Data containing only samples in both datasets, either without duplicates
            (set intersection) or taking duplicates into account (multiset intersection)

        Raises:
            NotImplementedError or InvalidParameterError
            if the intersection can not be computed
        """

        # the Data instance of _intersection will never be called
        # by set_intersection() as lhs and rhs can not be of type Data.

        raise NotImplementedError

    @staticmethod
    def _complement(lhs: "Data", rhs: "Data", duplicates: bool = False) -> "Data":
        """Specialized (multi)set complement.

        This complement method does not retain duplicates by default.
        For multiset behaviour, specify 'duplicates=True'.

        Derived classes can, but do not have to, override this method to
        provide an efficient specialized set complement method. Such
        methods should validate the type of both arguments, and, if they
        can not compute the set complement for them, raise NotImplementedError.

        Parameters:
            lhs: set A in A - B ('left hand side')
            rhs: set B in A - B ('right hand side')
            duplicates: if False (default), the returned data do not contain
                duplicate entries; if True, duplicates are taken into account.
                Both inputs and labels have to match for duplicates.

        Returns:
            Data containing all samples in lhs, but not in rhs, either without duplicates
            (set complement) or taking duplicates into account (multiset complement).

        Raises:
            NotImplementedError or InvalidParameterError
            if the set complement can not be computed
        """

        # the Data instance of _complement will never be called
        # by set_complement() as lhs and rhs can not be of type Data.

        raise NotImplementedError


# set operations


def intersection(lhs: "Data", rhs: "Data", duplicates: bool = False) -> "Data":
    """(Multi)set intersection of two datasets.

    This intersection method does not retain duplicates by default.
    For multiset behaviour, specify 'duplicates=True'.

    Parameters:
        lhs: one of the two datasets to intersect ('left hand side')
        rhs: one of the two datasets to intersect ('right hand side')
        duplicates: if False (default), the returned data do not contain
            duplicate entries; if True, duplicates are taken into account.
            Both inputs and labels have to match for duplicates.

    Returns:
        Data containing only samples in both datasets, without duplicates
    """

    # parameter validation
    lhs = params.instance(lhs, Data)
    rhs = params.instance(rhs, Data)

    # special case: empty set
    if lhs.num_samples == 0:
        return lhs.subset()
    if rhs.num_samples == 0:
        return rhs.subset()

    # try specialized implementations
    exception = None
    try:
        if hasattr(lhs.__class__, "_intersection"):
            return lhs.__class__._intersection(lhs, rhs, duplicates)
    except (NotImplementedError, InvalidParameterError) as e:
        exception = e

    try:
        if hasattr(rhs.__class__, "_intersection"):
            return rhs.__class__._intersection(lhs, rhs, duplicates)
    except (NotImplementedError, InvalidParameterError) as e:
        exception = e

    # no specialized method found or succeeded
    raise NotImplementedError("generalized (multi)set intersection not implemented") from exception


def complement(lhs: "Data", rhs: "Data", duplicates: bool = False) -> "Data":
    """(Multi)set complement of two datasets.

    This complement method does not retain duplicates by default.
    For multiset behaviour, specify 'duplicates=True'.

    Parameters:
        lhs: set A in A - B ('left hand side')
        rhs: set B in A - B ('right hand side')
        duplicates: if False (default), the returned data do not contain
            duplicate entries; if True, duplicates are taken into account.
            Both inputs and labels have to match for duplicates.

    Returns:
        Data containing all samples in lhs, but not in rhs, without duplicates
    """

    # parameter validation
    lhs = params.instance(lhs, Data)
    rhs = params.instance(rhs, Data)

    # special case: empty set
    if lhs.num_samples == 0:
        return lhs.subset()
    if rhs.num_samples == 0:
        return lhs.subset()

    # try specialized implementations
    exception = None
    try:
        if hasattr(lhs.__class__, "_complement"):
            return lhs.__class__._complement(lhs, rhs, duplicates)
    except (NotImplementedError, InvalidParameterError) as e:
        exception = e

    try:
        if hasattr(rhs.__class__, "_complement"):
            return rhs.__class__._complement(lhs, rhs, duplicates)
    except (NotImplementedError, InvalidParameterError) as e:
        exception = e

    # no specialized method found or succeeded
    raise NotImplementedError("generalized (multi)set complement not implemented") from exception
