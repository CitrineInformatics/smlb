"""VectorSpaceData tests.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp 2019-2020, Citrine Informatics.
"""

import pytest

import numpy as np

import smlb

from test_data import validate_data_interface


def test_VectorSpaceData_instantiation():
    """Test VectorSpaceData initialization"""

    # unlabeled, no domain
    ds = smlb.VectorSpaceData(dimensions=3)
    validate_data_interface(ds)
    assert ds.dimensions == 3 and ds.domain is None
    assert not ds.is_labeled and not ds.is_finite
    assert (ds.samples([[1, 2, 3], [4, 5, 6]]) == [[1, 2, 3], [4, 5, 6]]).all()
    with pytest.raises(smlb.BenchmarkError):
        ds.labels()

    # unlabeled, domain
    ds = smlb.VectorSpaceData(dimensions=3, domain=(1, 5))
    validate_data_interface(ds)
    assert ds.dimensions == 3 and len(ds.domain) == 3
    assert not ds.is_labeled and not ds.is_finite
    assert (ds.samples([[1, 2, 3], [4, 5, 5]]) == [[1, 2, 3], [4, 5, 5]]).all()
    with pytest.raises(smlb.InvalidParameterError):
        ds.samples([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(smlb.BenchmarkError):
        ds.labels()

    # labeled, no domain
    f = lambda arg: np.sum(arg, axis=1)  # noqa: E731
    ds = smlb.VectorSpaceData(dimensions=2, function=f)
    validate_data_interface(ds)
    assert ds.dimensions == 2 and ds.domain is None
    assert ds.is_labeled and not ds.is_finite
    assert (ds.labels([[1, 2], [3, 4]]) == [3, 7]).all()


@pytest.fixture
def fixture_VectorSpaceData_parabola_1d():
    f = lambda v: v[:, 0] ** 2  # noqa: E731
    ds = smlb.VectorSpaceData(dimensions=1, function=f, domain=[-2, 2])
    return ds


def test_VectorSpaceData_parabola_1d(fixture_VectorSpaceData_parabola_1d):
    """Simple parabola example"""

    ds = fixture_VectorSpaceData_parabola_1d
    assert ds.labels(((0.0,),)) == 0.0
    assert ds.labels([[2]]) == 4.0

    # outside of domain
    with pytest.raises(smlb.BenchmarkError):
        ds.labels([[np.nextafter(-2, -3)]])  # outside of domain to the left
    with pytest.raises(smlb.BenchmarkError):
        ds.labels([[np.nextafter(2, 3)]])  # outside of domain to the right


# def test_ComputedLabelsVectorSpaceData_intersection_1(
#     fixture_ComputedLabelsVectorSpaceData_parabola_1d,
# ):
#     """Test correctness of intersection for some examples.

#     Include test for different labels y of the same inputs x
#     being recognized as different examples (x,y).
#     """

#     ds = fixture_ComputedLabelsVectorSpaceData_parabola_1d
#     a = ds.subset([[-1], [0], [2], [1]])
#     b = ds.subset([[-1.1], [2], [0.5], [1]])

#     assert (np.sort(a.intersection(b).samples()) == [[1.0], [2.0]]).all()
#     assert (np.sort(a.intersection(b).labels()) == [1.0, 4.0]).all()

#     f = lambda v: v[:, 0] ** 2 + np.abs(v[:, 0])
#     ds2 = smlb.ComputedLabelsVectorSpaceData(dimensions=1, function=f, domain=[-2, 2])
#     c = ds2.subset([[1], [1.9], [0], [-0.5]])

#     assert (np.sort(a.intersection(c).samples()) == [[0.0]]).all()
#     assert (np.sort(a.intersection(c).labels()) == [0]).all()


# @pytest.mark.timeout(2)
# def test_ComputedLabelsVectorSpaceData_intersection_2(
#     fixture_ComputedLabelsVectorSpaceData_parabola_1d,
# ):
#     """Computational efficiency.

#     Full-match intersection for larger dataset with computed labels.

#     Because this test uses a parabola on [-2,2], it also tests
#     correctness of intersection for different inputs x with same
#     labels y in (x,y), e.g., (-1,1) and (1,1).
#     """

#     n = 5000
#     ds = fixture_ComputedLabelsVectorSpaceData_parabola_1d
#     inds = np.transpose(np.asfarray([np.linspace(-2, 2, n)]))
#     a = ds.subset(inds)
#     b = ds.subset(inds)

#     assert a.intersection(b).num_samples == n


# @pytest.mark.timeout(2)
# def test_intersection_2():
#     """Computational efficiency of intersection for an actual synthetic dataset as test case."""

#     from smlb.datasets.synthetic.friedman_1979.friedman_1979 import Friedman1979Data

#     data = Friedman1979Data(dimensions=10)

#     n = 1000
#     lhs = smlb.RandomVectorSampler(size=n, rng=0).fit(data).apply(data)
#     rhs = smlb.RandomVectorSampler(size=n, rng=1).fit(data).apply(data)
#     assert lhs.intersection(rhs).num_samples == 0
