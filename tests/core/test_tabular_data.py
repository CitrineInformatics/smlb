"""TabularData tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp 2019-2020, Citrine Informatics.
"""

import math
import timeit
from typing import Callable

import pytest

import numpy as np
import pandas as pd

import smlb

from test_data import validate_data_interface


#  #######################
#  #  Correctness tests  #
#  #######################


@pytest.fixture
def get_matrix_data():
    """Returns a function that creates various toy TabularData objects."""

    def create_TabularData(sel: int) -> smlb.TabularData:
        """Returns one of a given list of TabularData objects.

        TabularData objects are created anew on each invocation.
        Changes to them are not persistent.

        Parameter:
            sel: which TabularData object to return
                 0   empty set
                10   2 x 3, numerical unlabeled
                11   2 x 3, numerical scalar labels
                12   2 x 3, numerical vector labels
                13   2 x 3, mixed unlabeled
                14   2 x 3, mixed labeled
                20   7 x 3, numerical unlabeled
                21   7 x 3, numerical scalar labels
                22   7 x 3, numerical vector labels
                23   7 x 3, mixed unlabeled
                24   7 x 3, mixed labeled

        Returns:
            Selected TabularData object

        0: empty set

        10: 2x3, numerical unlabeled
            1 2 3
            4 5 6
        11: 2x3, numerical scalar labels
            1 2 3    10
            4 5 6    20
        12: 2x3, numerical vector labels
            1 2 3    10 11
            4 5 6    20 21
        13: 2x3, mixed unlabeled
            1 'a' 1.1
            2 'b' 2.2
        14: 2x3, mixed labeled
            1 'a' 1.1    'x', 10
            2 'b' 2.2    'y', 20

        20: 7x3, numerical unlabeled
            1 2 3.3    # 0
            4 5 6.6    # 1
            7 8 9.9    # 2
            1 2 3.3    # 3
            3 5 6.6    # 4
            7 8 9.9    # 5
            1 2 3.3    # 6
        21: 7x3, numerical scalar labels
            1 2 3.3    1    # 0
            4 5 6.6    2    # 1
            7 8 9.9    3    # 2
            1 2 3.3    1    # 3
            3 5 6.6    5    # 4
            7 8 9.9    3    # 5
            1 2 3.3    7    # 6
        22: 7x3, numerical vector labels
            1 2 3.3    1 11    # 0
            4 5 6.6    2 22    # 1
            7 8 9.9    3 33    # 2
            1 2 3.3    1 11    # 3
            3 5 6.6    5 55    # 4
            7 8 9.9    3 33    # 5
            1 2 3.3    7 77    # 6
        23: 7x3, mixed unlabeled
            1 'b' 3.3    # 0
            4 'e' 6.6    # 1
            7 'h' 9.9    # 2
            1 'b' 3.3    # 3
            3 'e' 6.6    # 4
            7 'h' 9.9    # 5
            1 'b' 3.3    # 6
        24: 7x3, mixed labeled
            1 'b' 3.3    'a' 11    # 0
            4 'e' 6.6    'b' 22    # 1
            7 'h' 9.9    'c' 33    # 2
            1 'b' 3.3    'a' 11    # 3
            3 'e' 6.6    'e' 55    # 4
            7 'h' 9.9    'c' 33    # 5
            1 'b' 3.3    'g' 77    # 6

        """

        data10 = np.asarray([[1, 2, 3], [4, 5, 6]])
        labels11 = np.asarray([10, 20])
        labels12 = np.asarray([[10, 11], [20, 21]])
        data13 = np.asarray(
            [(1, "a", 1.1), (2, "b", 2.2)], dtype=[("A", int), ("B", "U1"), ("C", float)]
        )
        labels14 = np.asarray([("x", 10), ("y", 20)], dtype=[("X", "U1"), ("Y", int)])

        data20 = np.asarray(
            [
                [1, 2, 3.3],  # 0
                [4, 5, 6.6],  # 1
                [7, 8, 9.9],  # 2
                [1, 2, 3.3],  # 3
                [3, 5, 6.6],  # 4
                [7, 8, 9.9],  # 5
                [1, 2, 3.3],  # 6
            ]
        )
        labels21 = np.array([1, 2, 3, 1, 5, 3, 7])
        labels22 = np.array([[1, 11], [2, 22], [3, 33], [1, 11], [5, 55], [3, 33], [7, 77]])

        data23 = np.array(
            [
                (1, "b", 3.3),  # 0
                (4, "e", 6.6),  # 1
                (7, "h", 9.9),  # 2
                (1, "b", 3.3),  # 3
                (3, "e", 6.6),  # 4
                (7, "h", 9.9),  # 5
                (1, "b", 3.3),  # 6
            ],
            dtype=[("A", int), ("B", "U1"), ("C", float)],
        )
        labels24 = np.array(
            [("a", 11), ("b", 22), ("c", 33), ("a", 11), ("e", 55), ("c", 33), ("g", 77)],
            dtype=[("X", "U1"), ("Y", int)],
        )

        if sel == 0:  # empty set
            return smlb.TabularData(data=np.empty(shape=(0, 0)))

        elif sel == 10:  # 2 x 3, unlabeled
            return smlb.TabularData(data=data10)
        elif sel == 11:  # 2 x 3, scalar labels
            return smlb.TabularData(data=data10, labels=labels11)
        elif sel == 12:  # 2 x 3, vector labels
            return smlb.TabularData(data=data10, labels=labels12)
        elif sel == 13:  # 2 x 3, mixed unlabeled
            return smlb.TabularData(data=data13)
        elif sel == 14:  # 2 x 3, mixed labeled
            return smlb.TabularData(data=data13, labels=labels14)
        elif sel == 20:  # 7 x 3, unlabeled, with repetitions
            return smlb.TabularData(data=data20)
        elif sel == 21:  # 7 x 3, scalar labels, with repetitions
            return smlb.TabularData(data=data20, labels=labels21)
        elif sel == 22:  # 7 x 3, vector labels, with repetitions
            return smlb.TabularData(data=data20, labels=labels22)
        elif sel == 23:  # 7 x 3, mixed unlabeled
            return smlb.TabularData(data=data23)
        elif sel == 24:  # 7 x 3, mixed labeled
            return smlb.TabularData(data=data23, labels=labels24)
        else:
            raise smlb.InvalidParameterError("dataset identifier", sel)

    create_TabularData.datasets = [0, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]

    return create_TabularData


def test_TabularData_fixture(get_matrix_data):
    """Validate Data interface for all pre-defined TabularData."""

    for sel in get_matrix_data.datasets:
        ds = get_matrix_data(sel)
        validate_data_interface(ds)


def test_TabularData_instantiation(get_matrix_data):
    """Tests basic functionality for examples."""

    # empty set
    ds = get_matrix_data(0)
    assert ds.is_finite and ds.num_samples == 0

    # numerical, unlabeled
    ds = get_matrix_data(10)
    assert (ds.samples() == [[1, 2, 3], [4, 5, 6]]).all()
    with pytest.raises(smlb.BenchmarkError):
        ds.labels()
    assert ds.num_samples == 2
    assert not ds.is_labeled and ds.is_finite

    # numerical, scalar labels
    ds = get_matrix_data(11)
    assert (ds.samples() == [[1, 2, 3], [4, 5, 6]]).all()
    assert (ds.labels() == [10, 20]).all()
    assert ds.num_samples == 2
    assert ds.is_labeled and ds.is_finite

    # numerical, vector labels
    ds = get_matrix_data(12)
    assert (ds.samples() == [[1, 2, 3], [4, 5, 6]]).all()
    assert (ds.labels() == [[10, 11], [20, 21]]).all()
    assert ds.num_samples == 2
    assert ds.is_labeled and ds.is_finite

    # mixed, unlabeled
    ds = get_matrix_data(13)
    assert (ds.samples() == np.asarray([(1, "a", 1.1), (2, "b", 2.2)], dtype=ds._data.dtype)).all()
    with pytest.raises(smlb.BenchmarkError):
        ds.labels()
    assert ds.num_samples == 2
    assert not ds.is_labeled and ds.is_finite

    # mixed, labeled
    ds = get_matrix_data(14)
    assert (ds.samples() == np.asarray([(1, "a", 1.1), (2, "b", 2.2)], dtype=ds._data.dtype)).all()
    assert (ds.labels() == np.asarray([("x", 10), ("y", 20)], dtype=ds._labels.dtype)).all()
    assert ds.num_samples == 2
    assert ds.is_labeled and ds.is_finite


def test_TabularData_subset_correctness(get_matrix_data):
    """Test TabularData.subset functionality"""

    # empty set
    ds = get_matrix_data(0)
    ss = ds.subset([])
    assert ss.is_finite and ss.num_samples == 0
    ss = ds.subset()
    assert ss.is_finite and ss.num_samples == 0

    # 2 x 3, all versions
    ds = get_matrix_data(10)
    ss = ds.subset([1])
    assert not ss.is_labeled and ss.is_finite
    assert (ss.samples() == [[4, 5, 6]]).all()

    ds = get_matrix_data(11)
    ss = ds.subset([1])
    assert ss.is_labeled and ss.is_finite
    assert (ss.samples() == [[4, 5, 6]]).all()
    assert (ss.labels() == np.asarray([20])).all()

    ds = get_matrix_data(12)
    ss = ds.subset([0])
    assert ss.is_labeled and ss.is_finite
    assert (ss.samples() == [[1, 2, 3]]).all()
    assert (ss.labels() == np.asarray([10, 11])).all()

    ds = get_matrix_data(13)
    ss = ds.subset([1])
    assert not ss.is_labeled and ss.is_finite
    assert (ss.samples() == np.asarray([(2, "b", 2.2)], dtype=ds._data.dtype)).all()

    ds = get_matrix_data(14)
    ss = ds.subset([1])
    assert ss.is_labeled and ss.is_finite
    assert (ss.samples() == np.asarray([(2, "b", 2.2)], dtype=ds._data.dtype)).all()
    assert (ss.labels() == np.asarray([("y", 20)], dtype=ds._labels.dtype)).all()

    # order
    ss10 = get_matrix_data(10).subset([1, 0])
    ss11 = get_matrix_data(11).subset([1, 0])
    ss12 = get_matrix_data(12).subset([1, 0])
    ss13 = get_matrix_data(13).subset([1, 0])
    ss14 = get_matrix_data(14).subset([1, 0])
    assert (ss10.samples() == [[4, 5, 6], [1, 2, 3]]).all()
    assert (ss11.samples() == ss10.samples()).all()
    assert (ss11.labels() == np.asarray([20, 10])).all()
    assert (ss12.samples() == ss10.samples()).all()
    assert (ss12.labels() == np.asarray([[20, 21], [10, 11]])).all()
    dt = ss13._data.dtype
    assert (ss13.samples() == np.asarray([(2, "b", 2.2), (1, "a", 1.1)], dtype=dt)).all()
    dtx, dty = ss14._data.dtype, ss14._labels.dtype
    assert (ss14.samples() == np.asarray([(2, "b", 2.2), (1, "a", 1.1)], dtype=dtx)).all()
    assert (ss14.labels() == np.asarray([("y", 20), ("x", 10)], dtype=dty)).all()

    ss20 = get_matrix_data(20).subset([0, 2])
    ss24 = get_matrix_data(24).subset([0, 2])
    assert (ss20.samples() == [[1, 2, 3.3], [7, 8, 9.9]]).all()
    assert (ss24.samples() == np.asarray([(1, "b", 3.3), (7, "h", 9.9)], dtype=dtx)).all()


def test_TabularData_set_intersection_correctness(get_matrix_data):
    """Tests for correctness of intersection for TabularData without duplicates."""

    # unlabeled

    lhs = get_matrix_data(20).subset([0, 1, 2, 3])
    rhs = get_matrix_data(20).subset([4, 5, 6])
    intersection = smlb.intersection(lhs, rhs)
    assert (intersection.samples() == [[1, 2, 3.3], [7, 8, 9.9]]).all()  # order

    # labeled (scalars)

    lhs = get_matrix_data(21).subset([0, 1, 2, 3])
    rhs = get_matrix_data(21).subset([4, 5, 6])
    intersection = smlb.intersection(lhs, rhs)
    assert (intersection.samples() == [[7, 8, 9.9]]).all()
    assert (intersection.labels() == [3]).all()

    # labeled (vectors)

    lhs = get_matrix_data(22).subset([0, 1, 2, 3])
    rhs = get_matrix_data(22).subset([4, 5, 6])
    intersection = smlb.intersection(lhs, rhs)
    assert (intersection.samples() == [[7, 8, 9.9]]).all()
    assert (intersection.labels() == [[3, 33]]).all()

    # mixed unlabeled

    lhs = get_matrix_data(23).subset([0, 1, 2, 3])
    rhs = get_matrix_data(23).subset([4, 5, 6])
    intersection = smlb.intersection(lhs, rhs)
    dt = lhs._data.dtype
    assert (intersection.samples() == np.array([(1, "b", 3.3), (7, "h", 9.9)], dtype=dt)).all()

    # mixed labeled

    lhs = get_matrix_data(24).subset([0, 1, 2, 3])
    rhs = get_matrix_data(24).subset([4, 5, 6])
    intersection = smlb.intersection(lhs, rhs)
    dt1, dt2 = lhs._data.dtype, lhs._labels.dtype
    assert (intersection.samples() == np.array([(7, "h", 9.9)], dtype=dt1)).all()
    assert (intersection.labels() == np.array([("c", 33)], dtype=dt2)).all()


def test_TabularData_set_complement_correctness(get_matrix_data):
    """Tests for correctness of complement for TabularData without duplicates."""

    # numerical unlabeled

    lhs = get_matrix_data(20).subset([0, 1, 2, 3])
    rhs = get_matrix_data(20).subset([4, 5, 6])
    complement = smlb.complement(lhs, rhs)
    assert (complement.samples() == [[4, 5, 6.6]]).all()

    # numerical labeled (scalars)

    lhs = get_matrix_data(21).subset([4, 5, 6])
    rhs = get_matrix_data(21).subset([3, 2, 1, 0])
    complement = smlb.complement(lhs, rhs)
    assert (complement.samples() == [[3, 5, 6.6], [1, 2, 3.3]]).all()
    assert (complement.labels() == [5, 7]).all()

    # numerical labeled (vectors)

    lhs = get_matrix_data(22).subset([4, 5, 6])
    rhs = get_matrix_data(22).subset([3, 2, 1, 0])
    complement = smlb.complement(lhs, rhs)
    assert (complement.samples() == [[3, 5, 6.6], [1, 2, 3.3]]).all()
    assert (complement.labels() == [[5, 55], [7, 77]]).all()

    # mixed unlabeled

    lhs = get_matrix_data(23).subset([4, 5, 6])
    rhs = get_matrix_data(23).subset([3, 2, 1, 0])
    complement = smlb.complement(lhs, rhs)
    dt = lhs._data.dtype
    assert (complement.samples() == np.array([(3, "e", 6.6)], dtype=dt)).all()

    # mixed labeled

    lhs = get_matrix_data(24).subset([4, 5, 6])
    rhs = get_matrix_data(24).subset([3, 2, 1, 0])
    complement = smlb.complement(lhs, rhs)
    dt1, dt2 = lhs._data.dtype, lhs._labels.dtype

    assert (complement.samples() == np.array([(3, "e", 6.6), (1, "b", 3.3)], dtype=dt1)).all()
    assert (complement.labels() == np.array([("e", 55), ("g", 77)], dtype=dt2)).all()


#  #######################
#  #  Performance tests  #
#  #######################


@pytest.fixture
def get_benchmark_data():
    """Function that returns a benchmarking function.

    Two datasets of size n and 2n.
    Two samples, one repeated twice, are hidden in the data to verify correctness.

    0: test function set subset
    1: test function multiset subset
    2: test function set intersection
    3: test function multiset intersection
    4: test function set complement
    5: test function multiset complement
    20: setup for numerical unlabeled data
    21: setup for numerical labeled data
    22: setup for mixed unlabeled data
    23: setup for mixed labeled data
    """

    import string

    def set_subset_test():
        """Run set complement code. Requires ds1 to have been set up."""

        subset = ds1.subset()
        assert subset.num_samples == ds1.num_samples - 1

    set_subset_test.name = "set subset"

    def set_intersection_test():
        """Run intersection code. Requires ds1, ds2 to have been set up."""

        intersection = smlb.intersection(ds1, ds2)
        assert intersection.num_samples == 2

    set_intersection_test.name = "set intersection"

    def set_complement_test():
        """Run complement test code. Requires ds1, ds2 to have been set up."""

        complement = smlb.complement(ds1, ds2)
        assert complement.num_samples == ds1.num_samples - 3

    set_complement_test.name = "set complement"

    def numeric_unlabeled_setup(n: int):
        global ds1, ds2

        data1 = np.random.uniform(size=(n, 3))
        data1[int(0.33 * n)] = [1, 2, 3]
        data1[int(0.83 * n)] = data1[int(0.01 * n)] = [4, 5, 6]  # duplicate
        ds1 = smlb.TabularData(data=data1)

        data2 = np.random.uniform(size=(2 * n, 3))
        data2[int(1.9 * n)] = [1, 2, 3]
        data2[int(0.5 * n)] = [4, 5, 6]
        ds2 = smlb.TabularData(data=data2)

    numeric_unlabeled_setup.name = "numeric unlabeled"

    def numeric_labeled_setup(n: int):
        global ds1, ds2

        data1 = np.random.uniform(size=(n, 3))
        labels1 = np.random.uniform(size=(n, 2))
        data1[int(0.33 * n)] = [1, 2, 3]
        labels1[int(0.33 * n)] = [11, 22]
        data1[int(0.83 * n)] = data1[int(0.01 * n)] = [4, 5, 6]  # duplicate
        labels1[int(0.83 * n)] = labels1[int(0.01 * n)] = [44, 55]  # duplicate
        ds1 = smlb.TabularData(data=data1, labels=labels1)

        data2 = np.random.uniform(size=(2 * n, 3))
        labels2 = np.random.uniform(size=(2 * n, 2))
        data2[int(1.9 * n)] = [1, 2, 3]
        labels2[int(1.9 * n)] = [11, 22]
        data2[int(0.5 * n)] = [4, 5, 6]
        labels2[int(0.5 * n)] = [44, 55]
        ds2 = smlb.TabularData(data=data2, labels=labels2)

    numeric_labeled_setup.name = "numeric labeled"

    def mixed_unlabeled_setup(n: int):
        global ds1, ds2

        dt = [("A", float), ("B", "U1"), ("C", float)]

        data1a = np.random.uniform(size=n)
        data1b = np.random.choice(list(string.ascii_letters), n)
        data1c = np.random.uniform(size=n)
        data1 = np.array([(a, b, c) for a, b, c in zip(data1a, data1b, data1c)], dtype=dt)
        data1[int(0.33 * n)] = (1, "b", 3)
        data1[int(0.83 * n)] = data1[int(0.01 * n)] = (4, "c", 6)  # duplicate
        ds1 = smlb.TabularData(data=data1)

        data2a = np.random.uniform(size=2 * n)
        data2b = np.random.choice(list(string.ascii_letters), 2 * n)
        data2c = np.random.uniform(size=2 * n)
        data2 = np.array([(a, b, c) for a, b, c in zip(data2a, data2b, data2c)], dtype=dt)
        data2[int(1.9 * n)] = (1, "b", 3)
        data2[int(0.5 * n)] = (4, "c", 6)
        ds2 = smlb.TabularData(data=data2)

    mixed_unlabeled_setup.name = "mixed unlabeled"

    def mixed_labeled_setup(n):
        global ds1, ds2

        dts = [("A", float), ("B", "U1"), ("C", float)]
        dtl = [("X", "U2"), ("Y", int)]

        data1a = np.random.uniform(size=n)
        data1b = np.random.choice(list(string.ascii_letters), n)
        data1c = np.random.uniform(size=n)
        data1 = np.array([(a, b, c) for a, b, c in zip(data1a, data1b, data1c)], dtype=dts)
        data1[int(0.33 * n)] = (1, "b", 3)
        data1[int(0.83 * n)] = data1[int(0.01 * n)] = (4, "c", 6)  # duplicate

        labels1a = np.random.choice(list(string.ascii_letters), n)
        labels1b = np.random.randint(32000, size=n)
        labels1 = np.array([(x, y) for x, y in zip(labels1a, labels1b)], dtype=dtl)
        labels1[int(0.33 * n)] = ("xx", 22)
        labels1[int(0.83 * n)] = labels1[int(0.01 * n)] = ("yy", 55)  # duplicate

        ds1 = smlb.TabularData(data=data1, labels=labels1)

        data2a = np.random.uniform(size=2 * n)
        data2b = np.random.choice(list(string.ascii_letters), 2 * n)
        data2c = np.random.uniform(size=2 * n)
        data2 = np.array([(a, b, c) for a, b, c in zip(data2a, data2b, data2c)], dtype=dts)
        data2[int(1.9 * n)] = (1, "b", 3)
        data2[int(0.5 * n)] = (4, "c", 6)

        labels2a = np.random.choice(list(string.ascii_letters), 2 * n)
        labels2b = np.random.randint(32000, size=2 * n)
        labels2 = np.array([(x, y) for x, y in zip(labels2a, labels2b)], dtype=dtl)
        labels2[int(1.9 * n)] = ("xx", 22)
        labels2[int(0.5 * n)] = ("yy", 55)

        ds2 = smlb.TabularData(data=data2, labels=labels2)

    mixed_labeled_setup.name = "mixed labeled"

    def create_benchmark_data(fn: int) -> Callable:
        if fn == 0:
            return set_subset_test
        elif fn == 1:
            raise NotImplementedError
        elif fn == 2:
            return set_intersection_test
        elif fn == 3:
            raise NotImplementedError
        elif fn == 4:
            return set_complement_test
        elif fn == 5:
            raise NotImplementedError
        elif fn == 20:
            return numeric_unlabeled_setup
        elif fn == 21:
            return numeric_labeled_setup
        elif fn == 22:
            return mixed_unlabeled_setup
        elif fn == 23:
            return mixed_labeled_setup
        else:
            raise smlb.InvalidParameterError("Function selector", fn)

    return create_benchmark_data


def run_benchmark(sizes, testfind, setupfinds, get_benchmark_data):
    """Runs benchmarks."""

    np.random.seed(83834)

    print()  # start output on own line
    testf = get_benchmark_data(testfind)
    for f in setupfinds:
        setupf = get_benchmark_data(f)
        print(f"TabularData [{setupf.name}, {testf.name}]:")
        for s in sizes:
            t = timeit.timeit(stmt=f"testf()", setup=f"setupf(s)", globals=locals(), number=1)
            print(f"log(s) = {round(math.log10(s),1):3.1f}: {round(t,1):5.1f} s")
        print()


@pytest.mark.timeout(5)
def test_TabularData_set_subset_performance(get_benchmark_data):
    """Test that specialized set subset is reasonably fast."""

    run_benchmark(
        sizes=[10_000],
        testfind=0,
        setupfinds=[20, 21, 22, 23],
        get_benchmark_data=get_benchmark_data,
    )


@pytest.mark.skip(reason="too slow to run every time")
def test_TabularData_set_subset_performance_benchmark(get_benchmark_data):
    """Run timings for TabularData set subset

    Run via
    `pytest --capture=tee-sys tests/test_tabular_data.py::test_TabularData_set_subset_performance_benchmark`
    """

    # MacBook 2018, 2.7 GHz

    # TabularData [numeric unlabeled, set subset]:
    # log(s) = 4.0:   0.0 s
    # log(s) = 5.0:   0.1 s
    # log(s) = 6.0:   1.3 s
    # log(s) = 6.3:   2.9 s

    # TabularData [numeric labeled, set subset]:
    # log(s) = 4.0:   0.0 s
    # log(s) = 5.0:   0.1 s
    # log(s) = 6.0:   1.5 s
    # log(s) = 6.3:   3.4 s

    # TabularData [mixed unlabeled, set subset]:
    # log(s) = 4.0:   0.0 s
    # log(s) = 5.0:   0.1 s
    # log(s) = 6.0:   1.5 s
    # log(s) = 6.3:   3.4 s

    # TabularData [mixed labeled, set subset]:
    # log(s) = 4.0:   0.0 s
    # log(s) = 5.0:   0.1 s
    # log(s) = 6.0:   1.8 s
    # log(s) = 6.3:   3.9 s

    run_benchmark(
        sizes=[10_000, 100_000, 1_000_000, 2_000_000],
        testfind=0,
        setupfinds=[20, 21, 22, 23],
        get_benchmark_data=get_benchmark_data,
    )


@pytest.mark.timeout(5)
def test_TabularData_set_intersection_performance(get_benchmark_data):
    """Test that specialized set intersection is reasonably fast."""

    run_benchmark(
        sizes=[10_000],
        testfind=2,
        setupfinds=[20, 21, 22, 23],
        get_benchmark_data=get_benchmark_data,
    )


@pytest.mark.skip(reason="too slow to run every time")
def test_TabularData_set_intersection_performance_benchmark(get_benchmark_data):
    """Run timings for TabularData set intersection

    Run via
    `pytest --capture=tee-sys tests/test_tabular_data.py::test_TabularData_set_intersection_performance_benchmark`
    """

    # MacBook 2018, 2.7 GHz

    # TabularData [numeric unlabeled]:
    # log(s) = 4.0:   0.0 s
    # log(s) = 5.0:   0.2 s
    # log(s) = 6.0:   3.1 s
    # log(s) = 6.3:   7.0 s

    # TabularData [numeric labeled]:
    # log(s) = 4.0:   0.1 s
    # log(s) = 5.0:   0.3 s
    # log(s) = 6.0:   3.8 s
    # log(s) = 6.3:   8.7 s

    # TabularData [mixed unlabeled]:
    # log(s) = 4.0:   0.0 s
    # log(s) = 5.0:   0.3 s
    # log(s) = 6.0:   3.8 s
    # log(s) = 6.3:   8.5 s

    # TabularData [mixed labeled]:
    # log(s) = 4.0:   0.0 s
    # log(s) = 5.0:   0.3 s
    # log(s) = 6.0:   4.5 s
    # log(s) = 6.3:  10.0 s

    run_benchmark(
        sizes=[10_000, 100_000, 1_000_000, 2_000_000],
        testfind=2,
        setupfinds=[20, 21, 22, 23],
        get_benchmark_data=get_benchmark_data,
    )


@pytest.mark.timeout(5)
def test_TabularData_set_complement_performance(get_benchmark_data):
    """Test that specialized set complement is reasonably fast."""

    run_benchmark(
        sizes=[10_000],
        testfind=4,
        setupfinds=[20, 21, 22, 23],
        get_benchmark_data=get_benchmark_data,
    )


@pytest.mark.skip("too slow to run every time")
def test_TabularData_set_complement_performance_benchmark(get_benchmark_data):
    """Run timings for TabularData set intersection

    Run via
    `pytest --capture=tee-sys tests/test_tabular_data.py::test_TabularData_set_complement_performance_benchmark`
    """

    # MacBook 2018, 2.7 GHz

    # TabularData [numeric unlabeled, set complement]:
    # log(s) = 4.0:   0.1 s
    # log(s) = 5.0:   0.6 s
    # log(s) = 6.0:   7.5 s
    # log(s) = 6.3:  15.6 s

    # TabularData [numeric labeled, set complement]:
    # log(s) = 4.0:   0.1 s
    # log(s) = 5.0:   0.7 s
    # log(s) = 6.0:   8.0 s
    # log(s) = 6.3:  17.2 s

    # TabularData [mixed unlabeled, set complement]:
    # log(s) = 4.0:   0.1 s
    # log(s) = 5.0:   0.7 s
    # log(s) = 6.0:   8.2 s
    # log(s) = 6.3:  17.1 s

    # TabularData [mixed labeled, set complement]:
    # log(s) = 4.0:   0.1 s
    # log(s) = 5.0:   0.7 s
    # log(s) = 6.0:   9.0 s
    # log(s) = 6.3:  18.9 s

    run_benchmark(
        sizes=[10_000, 100_000, 1_000_000, 2_000_000],
        testfind=4,
        setupfinds=[20, 21, 22, 23],
        get_benchmark_data=get_benchmark_data,
    )


#  ###########################
#  #  TabularDataFromPandas  #
#  ###########################


def test_TabularDataFromPandas_initialization():
    """Test initializer"""

    # unlabeled
    data, columns = [[1, 2.0, "a"], [3, 4.0, "b"], [5, 6.0, "c"]], ["A", "B", "C"]
    df = pd.DataFrame(data, columns=columns)
    ds = smlb.TabularDataFromPandas(data=df)
    assert ds.is_finite and not ds.is_labeled and ds.num_samples == 3
    samples = np.array([tuple(row) for row in data], dtype=list(zip(columns, [int, float, "O"])))
    assert (ds.samples() == samples).all()

    # labeled (via column names)
    data, columns = [[1, 2.0, "a"], [3, 4.0, "b"], [5, 6.0, "c"]], ["A", "B", "C"]
    df = pd.DataFrame(data, columns=columns)
    ds = smlb.TabularDataFromPandas(data=df, labels=["B"])
    assert ds.is_finite and ds.is_labeled and ds.num_samples == 3
    samples = np.array([tuple([row[0], row[2]]) for row in data], dtype=[("A", int), ("C", "O")])
    labels = np.array([tuple([row[1]]) for row in data], dtype=[("B", float)])
    assert (ds.samples() == samples).all()
    assert (ds.labels() == labels).all()

    # labeled (via separate DataFrame)
    data_samples, columns_samples = [[1, 2.0, "a"], [3, 4.0, "b"], [5, 6.0, "c"]], ["A", "B", "C"]
    data_labels, columns_labels = [[9.0, "ab"], [10, "xy"], [11, ""]], ["X", "Y"]
    df_samples = pd.DataFrame(data_samples, columns=columns_samples)
    df_labels = pd.DataFrame(data_labels, columns=columns_labels)
    ds = smlb.TabularDataFromPandas(data=df_samples, labels=df_labels, dtype={"Y": "U2"})
    assert ds.is_finite and ds.is_labeled and ds.num_samples == 3
    samples = np.array(
        [tuple(row) for row in data_samples], dtype=list(zip(columns_samples, [int, float, "O"]))
    )
    labels = np.array(
        [tuple(row) for row in data_labels], dtype=list(zip(columns_labels, [float, "U2"]))
    )
    assert (ds.samples() == samples).all()
    assert (ds.labels() == labels).all()
