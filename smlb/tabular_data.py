"""TabularData.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp 2019-2020, Citrine Informatics.

Tabular indexed finite data.
Can handle several million data points.
See TabularData for details and tests for timings.
"""

from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from smlb.utility import is_sequence
from smlb.exceptions import BenchmarkError, InvalidParameterError
from smlb.parameters import params
from smlb.data import Data


# An alternative design to having a single TabularData class would be to have
# two classes, say MatrixData for purely numerical data, backed by NumPy, and
# IndexedFiniteData for arbitrary tabular data, backed by Pandas.
#
# While NumPy states in its documentation of structured arrays at
# https://numpy.org/doc/stable/user/basics.rec.html that
# "tabular data, such as stored in csv files, may find other pydata projects
# more suitable, such as xarray, pandas, or DataArray. These [...] are better
# optimized for that use. For instance, the C-struct-like memory layout of
# structured arrays in numpy can lead to poor cache behavior in comparison."
# those turn out to be sufficient in practice while avoiding the need to
# maintain two classes. For numerical data, the NumPy backend is much faster
# than a Pandas backend.

# todo: order in _intersection and _complement


class TabularData(Data):
    r"""Tabular finite indexed data.

    Both samples and labels are 0-based indexed arrays
    that can have zero or more further dimensions.
    Samples have two dimensions, labels can be scalars or vectors.

    Implemented using NumPy as backend.
    Efficient enough to handle several million samples.
    See unit tests for timings.
    """

    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None, **kwargs):
        """Initialize dataset.

        Parameters:
            data: tabular data as a NumPy ndarray
            labels: tabular data as a NumPy ndarray. If not specified,
                dataset is unlabeled.

        Raises:
            InvalidParameterError for invalid arguments. In particular,
                numbers of data and labels must match.

        Examples:
            From numerical NumPy data:
            ```
            TabularData(numpy.ndarray(...), ...)
            ```

            From a Pandas DataFrame:
            ```
            df = pandas.DataFrame(..., columns=[...])
            TabularData(df.to_records(index=False), labels=...)
            ```

            From mixed NumPy data, with column names (note use of tuples):
            ```
            a = numpy.array([('a', 1), ('b', 2)], dtype=[('C', str), ('D', int)])
            TabularData(a, ...)
            ```
        """

        # parameter validation
        data = params.instance(data, np.ndarray)
        labels = params.optional_(labels, lambda arg: params.instance(arg, np.ndarray))

        if labels is not None:
            # number of samples and labels must match
            if data.shape[0] != labels.shape[0]:
                raise InvalidParameterError(
                    "same number of samples and labels",
                    f"{data.shape[0]} samples, {labels.shape[0]} labels",
                )

            # uniqueness of "column" names, if any, is enforced by NumPy,
            # but only separately for data and labels
            if is_sequence(data.dtype.names) and is_sequence(labels.dtype.names):
                column_names = data.dtype.names + labels.dtype.names
                if len(column_names) != len(np.unique(column_names)):
                    raise InvalidParameterError(
                        "unique column names for samples and labels", column_names
                    )

        self._data, self._labels = data, labels

        super().__init__(**kwargs)

    @property
    def is_finite(self) -> bool:
        """Query for whether the data is finite or infinite.

        Returns:
            True
        """

        return True

    # testing function for indices arguments used by samples, subset and other functions
    def _indices_testf(self, indices: Sequence[Any]):
        return params.optional_(
            indices,
            lambda arg: list(
                params.any_(  # NumPy indexing expects a list
                    arg,
                    lambda arg: params.tuple_(arg, None, arity=0),  # empty set
                    lambda arg: params.tuple_(
                        arg, lambda arg: params.integer(arg, from_=0, below=self.num_samples)
                    ),
                )
            ),
        )

    def samples(self, indices: Optional[Sequence[int]] = None) -> np.ndarray:
        """Query samples.

        Returns a sequence of samples or raises InvalidParameterError.

        Parameters:
            indices: A sequence of non-negative integers in the range [0, n),
                where n is number of samples. By default, all samples are returned.

        Returns:
            NumPy ndarray of queried samples. Type of samples depends on the data.

        Raises:
            InvalidParameterError: for invalid keys
        """

        indices = self._indices_testf(indices)

        return self._data[indices] if indices else self._data

    @property
    def num_samples(self) -> int:
        """Query number of samples.

        Returns:
            Number of samples in data; guaranteed to be a non-negative integer.
        """

        return len(self._data)

    def labels(self, indices: Optional[Sequence[int]] = None) -> np.ndarray:
        """Query labels.

        Returns a sequence of labels or raises InvalidParameterError.

        Parameters:
            indices: A sequence of non-negative integers in the range [0, n),
                where n is number of samples. By default, all labels are returned.

        Returns:
            NumPy ndarray of labels. Type of labels depends on the data.

        Raises:
            InvalidParameterError: for invalid indices, or if dataset is not labeled
        """

        if not self.is_labeled:
            raise BenchmarkError("Querying labels of unlabeled data")

        indices = self._indices_testf(indices)

        return self._labels[indices] if indices else self._labels

    @property
    def is_labeled(self) -> bool:
        """Query for whether the data are labeled.

        Returns:
            True if data are labeled, False otherwise.
        """

        return self._labels is not None

    @staticmethod
    def _joint_data_labels(ds):
        """Single structured array for data and labels for comparison.

        Structured arrays can be used to run NumPy set methods
        on arrays with more than one dimension.
        """

        ds = params.instance(ds, TabularData)

        if is_sequence(ds._data.dtype.names):  # structured array
            lhs = ds._data
        else:  # homogeneous array, possibly many dimensions
            lhs = np.reshape(ds._data, (ds.num_samples, -1))
            lhs = lhs.view([("", ds._data.dtype)] * np.prod(lhs.shape[1:]))
            lhs = np.reshape(lhs, ds.num_samples)

        if not ds.is_labeled:
            result = lhs
        else:  # is_labeled
            # alternatives for hstack() that did not work included
            # numpy.lib.recfunctions.merge_arrays.

            if is_sequence(ds._labels.dtype.names):  # structured array
                rhs = ds._labels
            else:  # homogeneous array, possibly high-dimensional
                rhs = np.reshape(ds._labels, (ds.num_samples, -1))
                rhs = rhs.view([(str(i), rhs.dtype) for i in range(np.prod(rhs.shape[1:]))])
                rhs = np.reshape(rhs, ds.num_samples)

            # lhs and rhs are structured array (views) now
            # unfortunately, np.hstack fails for these
            dtypes = lhs.dtype.descr + rhs.dtype.descr
            result = np.empty(ds.num_samples, dtype=dtypes)
            for name in lhs.dtype.names:
                result[name] = lhs[name]
            for name in rhs.dtype.names:
                result[name] = rhs[name]

        return result

    def subset(
        self, indices: Optional[Sequence[int]] = None, duplicates: bool = False
    ) -> "TabularData":
        """Create finite subset of data.

        Parameters:
            indices: A sequence of non-negative integers in the range [0, n),
                where n is number of samples. If no indices are specified,
                the whole dataset is returned.
            duplicates: if False (default), the returned subset does not contain
                duplicate entries; if True, duplicates are kept. Both inputs
                and labels have to match for duplicates.

        Returns:
            TabularData that contains only the specified samples.

        If duplicates are dropped, the first occurrence is kept.
        """

        # validate parameters
        indices = self._indices_testf(indices)
        duplicates = params.boolean(duplicates)

        # special case: empty set
        if self.num_samples == 0:
            if indices is not None and len(indices) > 0:
                raise InvalidParameterError("empty indices", indices, "indices into empty set")
            empty = np.empty(shape=(0,) + self._data.shape[1:], dtype=self._data.dtype)
            return TabularData(data=empty, labels=[] if self.is_labeled else None)

        # special case: empty subset
        if indices is not None and len(indices) == 0:
            empty = np.empty(shape=(0,) + self._data.shape[1:], dtype=self._data.dtype)
            return TabularData(data=empty, labels=np.array([]) if self.is_labeled else None)

        # default is to return the whole set
        if indices is None:
            indices = ...  # Ellipsis

        # create subset data and labels
        subset = TabularData(
            data=self._data[indices], labels=self._labels[indices] if self.is_labeled else None
        )

        # remove duplicates if required
        if duplicates is False:
            joint = self._joint_data_labels(subset)
            _, unique = np.unique(joint, return_index=True, axis=0)
            if len(unique) != subset.num_samples:  # only do work if there are any duplicates
                unique = np.sort(unique)  # restores original order
                subset = TabularData(
                    data=subset._data[unique],
                    labels=subset._labels[unique] if subset.is_labeled else None,
                )
        # else: pass

        return subset

    @staticmethod
    def _intersection(
        lhs: "TabularData", rhs: "TabularData", duplicates: bool = False
    ) -> "TabularData":
        """Specialized intersection.

        For labeled data, labels are compared as well.

        The datasets must be compatible in the sense that both are of type
        TabularData or derived, and either labeled or unlabeled.

        Parameters:
            lhs: one of the two datasets to intersect ('left hand side')
            rhs: one of the two datasets to intersect ('right hand side')
            duplicates: if False (default), the returned data do not contain
                duplicate entries; if True, duplicates are taken into account.
                Both inputs and labels have to match for duplicates.

        Returns:
            TabularData containing only samples in both datasets, either without duplicates
            (set intersection) or taking duplicates into account (multiset intersection)

        Raises:
            NotImplementedError if the set intersection can not be computed
        """

        # parameter validation
        lhs = params.instance(lhs, TabularData)
        rhs = params.instance(rhs, TabularData)
        duplicates = params.boolean(duplicates)

        # special case: empty set
        if lhs.num_samples == 0:
            return lhs.subset()  # copy
        if rhs.num_samples == 0:
            return rhs.subset()  # copy

        if lhs.is_labeled != rhs.is_labeled:
            raise InvalidParameterError("compatible TabularData", "mismatch in labeling")

        # intersection calculation
        _lhs, _rhs = TabularData._joint_data_labels(lhs), TabularData._joint_data_labels(rhs)

        if _lhs.dtype != _rhs.dtype:
            raise InvalidParameterError(
                "Matching TabularData", f"{_lhs.dtype.descr} and {_rhs.dtype.descr}"
            )

        if duplicates is False:
            _, indices, _ = np.intersect1d(_lhs, _rhs, return_indices=True)  # drops any duplicates
            indices = np.sort(indices)  # restores original order
            return lhs.subset(indices)
        else:  # duplicates = True
            raise NotImplementedError(  # todo: implement
                "specialized multiset intersection not implemented for TabularData"
            )

    @staticmethod
    def _complement(
        lhs: "TabularData", rhs: "TabularData", duplicates: bool = False
    ) -> "TabularData":
        """Specialized (multi)set complement.

        For labeled data, labels are compared as well.

        The datasets must be compatible in the sense that both are of type
        DataMatrix or derived, and either labeled or unlabeled.

        Parameters:
            lhs: set A in A - B ('left hand side')
            rhs: set B in A - B ('right hand side')
            duplicates: if False (default), the returned data do not contain
                duplicate entries; if True, duplicates are taken into account.
                Both inputs and labels have to match for duplicates.

        Returns:
            Data containing all samples in lhs, but not in rhs, either without duplicates
            (set complement) or taking duplicates into account (multiset complement).
        """

        # parameter validation
        lhs = params.instance(lhs, TabularData)
        rhs = params.instance(rhs, TabularData)
        duplicates = params.boolean(duplicates)

        # special case: empty set
        if lhs.num_samples == 0:
            return lhs.subset()
        if rhs.num_samples == 0:
            return lhs.subset()

        if lhs.is_labeled != rhs.is_labeled:
            raise InvalidParameterError("compatible TabularData", "mismatch in labeling")

        # complement calculation
        _lhs, _rhs = TabularData._joint_data_labels(lhs), TabularData._joint_data_labels(rhs)

        if _lhs.dtype != _rhs.dtype:
            raise InvalidParameterError(
                "Matching TabularData", f"{_lhs.dtype.descr} and {_rhs.dtype.descr}"
            )

        if duplicates is False:
            # np.setdiff1d does not return indices, so we don't use it

            indices = np.arange(_lhs.size)[np.isin(_lhs, _rhs, invert=True)]  # indexes into _lhs
            _, indices2 = np.unique(_lhs[indices], return_index=True)  # indexes into indices
            indices = indices[np.sort(indices2)]  # restores order

            return lhs.subset(indices)

            # below implementation is correct but a bit slower:

            # # remove duplicates from _lhs
            # _, indices = np.unique(_lhs, return_index=True)
            # indices = np.sort(indices)  # restores original order
            # _lhs = _lhs[indices]

            # # remove any element from _rhs
            # _, indices, _ = np.intersect1d(_lhs, _rhs, return_indices=True)
            # indices = np.setdiff1d(np.arange(_lhs.size), indices, assume_unique=True)
        else:  # duplicates = True
            raise NotImplementedError(  # todo: implement
                "specialized multiset complement not implemented for TabularData"
            )


class TabularDataFromPandas(TabularData):
    """Base class for tabular finite indexed data.

    Helps with implementing new datasets:
    Derive from this class and implement custom loading in __init__,
    which should then call this class's initializer via `super().__init__(...)`.

    See some of the experimental datasets that come with smlb for examples.
    """

    def __init__(
        self,
        data: "pandas.DataFrame",  # noqa F821
        labels: Optional[Union["pandas.DataFrame", Sequence[str]]] = None,
        dtype: Optional[dict] = None,
        join: Optional[str] = None,
        filterf: Optional[Callable[[Any], bool]] = None,
        samplef: Optional[Callable[[Any], Any]] = None,
        labelf: Optional[Callable[[Any], Any]] = None,
        **kwargs,
    ):
        """Initialize dataset.

        Parameters control loading and preprocessing of the data. Order:
        1. joining
        2. filtering
        3. sample and label transform

        Parameters:
            data: the samples in the form of a Pandas DataFrame.
            labels: the labels, either in the form of a Pandas DataFrame with same number of rows
                as data and different column names, or in the form of a list of column names,
                which are then split out from the data and used as labels. If not specified,
                the dataset is unlabeled.
            dtype: the NumPy data types to use for samples and labels, in the form of a dictionary
                with column names as keys and dtypes as values. Can be used to override dtype
                auto-detection for some or all columns.
            join: if specified, name of "column" to join by; this changes labels
                to be sequences of single-entry labels
            filterf: a function that accepts a sample and returns whether to keep it
                (True) or exclude it (False). Default retains all samples
            samplef: function accepting and returning a sample; applied to all samples
                as post-processing
            labelf: function accepting and returning a label; applied to all labels
                as post-processing

        Raises:
            InvalidParameterError for invalid arguments. In particular,
                numbers of data and labels must match. If column names are given,
                they must be unique across data and labels, if any.
        """

        import pandas as pd  # only import if class is used

        # parameter validation
        data = params.instance(data, pd.DataFrame)
        labels = params.optional_(
            labels,
            lambda arg: params.any_(
                arg,
                lambda arg: params.instance(arg, pd.DataFrame),  # before tuple_
                lambda arg: params.tuple_(arg, params.string),
            ),
        )
        dtype = params.optional_(dtype, lambda arg: params.instance(arg, dict), default={})
        join = params.optional_(join, params.string)
        singleargf = lambda arg: params.callable(arg, num_pos_or_kw=1)  # noqa: E731
        filterf = params.optional_(filterf, singleargf)
        samplef = params.optional_(samplef, singleargf)
        labelf = params.optional_(labelf, singleargf)

        if labels is None and labelf:
            raise InvalidParameterError(
                "matching labels and label function", "label function specified for unlabeled data"
            )

        # process data
        data = data.reset_index(drop=True)

        # if labels are given as separate DataFrame, join them
        if isinstance(labels, pd.DataFrame):
            if len(data) != len(labels):
                raise InvalidParameterError(
                    "matching data and labelsa",
                    f"different number of rows ({len(data)} != {len(labels)})",
                )

            labels = labels.reset_index(drop=True)

            col_names = np.hstack((data.columns.values, labels.columns.values))
            if len(col_names) != len(np.unique(col_names)):
                raise InvalidParameterError(
                    "unique column names", f"{data.columns.values} and {labels.columns.values}"
                )

            data = pd.concat([data, labels], axis=1)
            labels = labels.columns.values

        # 1. optional joining
        if join:
            groups = data.groupby(join, sort=False, as_index=False)
            data = groups.aggregate(lambda tdf: tdf.tolist())

        # 2. optional filtering
        if filterf:
            selection = data.apply(filterf, axis=1)
            data = data[selection]

        # split data and labels
        if labels is not None:
            # DataFrame column indexing requires list, not tuple
            data, labels = data.drop(columns=list(labels)), data[list(labels)]

        # 3. optional sample and label transform
        if samplef:
            data = data.apply(samplef, axis=1, result_type="reduce")
            if isinstance(data, pd.Series):
                data = pd.DataFrame(data, columns=["Samples"])
        if labelf:
            labels = labels.apply(labelf, axis=1, result_type="reduce")
            if isinstance(labels, pd.Series):
                labels = pd.DataFrame(labels, columns=["Labels"])

        # convert to NumPy structured array
        data = self._to_numpy(data, dtype=dtype)
        labels = self._to_numpy(labels, dtype=dtype) if labels is not None else None

        super().__init__(data=data, labels=labels, **kwargs)

    def _to_numpy(self, df, dtype={}):
        """Convert Pandas DataFrame to NumPy ndarray.

        Parameters:
            df: Pandas DataFrame with named columns
            dtype: dtype for some or all columns
        """

        dtypes = []
        for col in df.columns:
            if col in dtype:
                dtypes.append((col, dtype[col]))
            else:
                dtypes.append((col, df[col].dtype.str))

        if df.shape[1] == 1:
            result = np.array([(row,) for row in df.iloc[:, 0]], dtype=dtypes)
        else:
            result = np.array([row for row in df.itertuples(index=False, name=None)], dtype=dtypes)

        return result
