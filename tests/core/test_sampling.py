"""Sampling tests.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
2019-2020, Citrine Informatics.
"""

import pytest

import decimal

import numpy as np

import smlb
from smlb import params

###############
#  setdiff2d  #
###############

# this is a helper function for testing purposes
# if it proves useful in another context, it could
# be moved to utility.py


def setdiff2d(a, b):
    """Set difference between two sets of vectors.

    Returns the unique vectors in a that are not in b.
    The equivalent of np.setdiff1d for matrix rows.

    Parameters:
        a: NumPy 2d array (row vectors)
        b: NumPy 2d array (row vectors)
    Returns:
        NumPy 2d array, rows are unique vectors from a that are not in b
    """

    a = params.real_matrix(a)
    b = params.real_matrix(b)

    # special cases
    if len(a) == 0:
        return np.zeros((0, 0), dtype=float)

    a_rows = a.view([("", a.dtype)] * a.shape[1])
    b_rows = b.view([("", b.dtype)] * b.shape[1])
    res = np.setdiff1d(a_rows, b_rows).view(a.dtype).reshape(-1, a.shape[1])

    return res


def test_setdiff2d():
    a = []
    b = []
    assert setdiff2d(a, b).size == 0  # all() for an empty array is ambiguous

    a = []
    b = [[4, 5, 6]]
    assert setdiff2d(a, b).size == 0

    a = [[4, 5], [1, 2]]
    b = [[1, 5]]
    assert (setdiff2d(a, b) == [[1, 2], [4, 5]]).all()

    a = [[1, 2], [3, 4]]
    b = [[1, 2], [3, 4]]
    assert setdiff2d(a, b).size == 0

    a = [[1, 2, 3], [4, 5, 6]]
    b = [[4, 5, 6], [7, 8, 9]]
    assert (setdiff2d(a, b) == [[1, 2, 3]]).all()

    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8, 9], [1, 2, 3], [11, 12, 13]]
    assert (setdiff2d(a, b) == [[4, 5, 6]]).all()


#########################
#  RandomSubsetSampler  #
#########################


@pytest.fixture
def fixture_RandomSubsetSampler_smallset():
    """Dataset and RandomSubsetSampler for small finite dataset."""

    def _create_ds_ss(size=10, draw=3, labels=False, rng=0):
        data = np.arange(size)
        dataset = smlb.TabularData(data=data, labels=np.arange(size) + 1 if labels else None)
        sampler = smlb.RandomSubsetSampler(size=draw, rng=rng)
        return dataset, sampler

    return _create_ds_ss


def test_RandomSubsetSampler_examples_correctness(fixture_RandomSubsetSampler_smallset):
    """RandomSubsetSampler, test cases related to correctness of sampled subsets."""

    # no labels
    ds, ss = fixture_RandomSubsetSampler_smallset(size=10, draw=3)
    r = ss.fit(ds).apply(ds)
    assert len(r.samples()) == 3 and len(np.unique(r.samples())) == 3
    assert all([0 <= s < 10 for s in r.samples()])

    # with labels
    ds, ss = fixture_RandomSubsetSampler_smallset(size=10, draw=3, labels=True)
    r = ss.fit(ds).apply(ds)
    assert len(r.samples()) == 3 and len(np.unique(r.samples())) == 3
    assert len(r.labels()) == 3 and len(np.unique(r.labels())) == 3
    assert all([0 <= s < 10 for s in r.samples()])
    assert all([0 < lbl <= 103 for lbl in r.labels()])


def test_RandomSubsetSampler_examples_rng(fixture_RandomSubsetSampler_smallset):
    """RandomSubsetSampler, test cases related to randomness."""

    # reproducible, no labels
    ds1, ss1 = fixture_RandomSubsetSampler_smallset(size=10, draw=5, rng=0)
    ds2, ss2 = fixture_RandomSubsetSampler_smallset(size=10, draw=5, rng=0)
    r1, r2 = ss1.fit(ds1).apply(ds1), ss2.fit(ds2).apply(ds2)
    assert (r1.samples() == r2.samples()).all()

    # results depend on rng, no labels
    ds1, ss1 = fixture_RandomSubsetSampler_smallset(size=10, draw=5, rng=0)
    ds2, ss2 = fixture_RandomSubsetSampler_smallset(size=10, draw=5, rng=1)
    r1, r2 = ss1.fit(ds1).apply(ds1), ss2.fit(ds2).apply(ds2)
    assert (r1.samples() != r2.samples()).any()

    # reproducible, with labels
    ds1, ss1 = fixture_RandomSubsetSampler_smallset(size=10, draw=5, labels=True, rng=0)
    ds2, ss2 = fixture_RandomSubsetSampler_smallset(size=10, draw=5, labels=True, rng=0)
    r1, r2 = ss1.fit(ds1).apply(ds1), ss2.fit(ds2).apply(ds2)
    assert (r1.samples() == r2.samples()).all() and (r1.labels() == r2.labels()).all()

    # results depend on rng, with labels
    ds1, ss1 = fixture_RandomSubsetSampler_smallset(size=10, draw=5, labels=True, rng=0)
    ds2, ss2 = fixture_RandomSubsetSampler_smallset(size=10, draw=5, labels=True, rng=1)
    r1, r2 = ss1.fit(ds1).apply(ds1), ss2.fit(ds2).apply(ds2)
    assert (r1.samples() != r2.samples()).any() and (r1.labels() != r2.labels()).any()


#########################
#  RandomVectorSampler  #
#########################


def test_RandomVectorSampler_1():
    """Simple test cases."""

    # without labels
    ds = smlb.VectorSpaceData(dimensions=2)
    ss = smlb.RandomVectorSampler(size=30, rng=0, domain=[[0, 1], [-3, 2]]).fit(ds).apply(ds)
    assert ss.samples().shape == (30, 2)
    assert (ss.samples()[:, 0] >= 0).all() and (ss.samples()[:, 1] >= -3).all()
    assert (ss.samples()[:, 0] <= 1).all() and (ss.samples()[:, 1] <= 2).all()

    ss = smlb.RandomVectorSampler(size=3, rng=0, domain=[[0, 1], [0, 1]]).fit(ds).apply(ds)
    ss2 = smlb.RandomVectorSampler(size=3, rng=0).fit(ds).apply(ds)
    assert ss2.samples().shape == (3, 2)
    assert (ss.samples() == ss2.samples()).all()

    ss1 = smlb.RandomVectorSampler(size=3, rng=0).fit(ds).apply(ds)
    ss2 = smlb.RandomVectorSampler(size=3, rng=0).fit(ds).apply(ds)
    assert (ss1.samples() == ss2.samples()).all()

    ss3 = smlb.RandomVectorSampler(size=3, rng=1).fit(ds).apply(ds)
    assert (ss1.samples() != ss3.samples()).any()

    # with labels
    with pytest.raises(smlb.BenchmarkError):
        ds = smlb.VectorSpaceData(
            dimensions=2, function=lambda x: np.sum(x, axis=1), domain=[-np.inf, np.inf]
        )
        smlb.RandomVectorSampler(size=3, rng=0).fit(ds).apply(ds)

    ds = smlb.VectorSpaceData(dimensions=2, function=lambda x: np.sum(x, axis=1), domain=[0, 1])
    ss = smlb.RandomVectorSampler(size=3, rng=0).fit(ds).apply(ds)
    assert ss.samples().shape == (3, 2)

    ss1 = smlb.RandomVectorSampler(size=3, rng=0).fit(ds).apply(ds)
    ss2 = smlb.RandomVectorSampler(size=3, rng=0).fit(ds).apply(ds)
    assert (ss1.samples() == ss2.samples()).all() and (ss1.labels() == ss2.labels()).all()

    ss3 = smlb.RandomVectorSampler(size=3, rng=1).fit(ds).apply(ds)
    assert (ss1.samples() != ss3.samples()).any()


#################
#  GridSampler  #
#################


@pytest.fixture
def fixture_GridSampler_parabola():
    """Dataset and GridSampler for parabola in k dimensions."""

    def _create_ds_ss(dim, size=0, rng=0):
        f = lambda v: np.sum(np.power(v, 2), axis=1)  # noqa E731
        dataset = smlb.VectorSpaceData(dim, f, [-2, 2])
        sampler = smlb.GridSampler(size, rng=rng)
        return dataset, sampler

    return _create_ds_ss


def test_GridSampler_next_grid_size(fixture_GridSampler_parabola):
    """Grid-size helper function: check special cases."""

    ds, ss = fixture_GridSampler_parabola(dim=5)
    assert ss.next_grid_size(ds, 1) == 1
    assert ss.next_grid_size(ds, 2) == 2
    assert ss.next_grid_size(ds, 32) == 2
    assert ss.next_grid_size(ds, 33) == 3
    assert ss.next_grid_size(ds, 243) == 3
    assert ss.next_grid_size(ds, 244) == 4
    assert ss.next_grid_size(ds, 1024) == 4
    assert ss.next_grid_size(ds, 1025) == 5
    assert ss.next_grid_size(ds, 3125) == 5
    assert ss.next_grid_size(ds, 3126) == 6
    assert ss.next_grid_size(ds, 7776) == 6
    assert ss.next_grid_size(ds, 7777) == 7


def test_GridSampler_next_grid_size_2(fixture_GridSampler_parabola):
    """Grid-size helper function: compare to alternative correct but slow implementation."""

    ds, ss = fixture_GridSampler_parabola(3)

    def alt_impl(n, d):
        k = decimal.Decimal(n) ** (decimal.Decimal(1) / decimal.Decimal(d))
        return int(k.to_integral_exact(rounding=decimal.ROUND_CEILING))

    for i in range(1, 2500):  # too slow for larger n
        assert ss.next_grid_size(ds, i) == alt_impl(i, 3)


def test_GridSampler_next_grid_size_3(fixture_GridSampler_parabola):
    """Grid-size helper function: test limiting case for dimensionality."""

    ds, ss = fixture_GridSampler_parabola(1)
    assert ss.next_grid_size(ds, 1) == 1
    assert ss.next_grid_size(ds, 2) == 2
    assert ss.next_grid_size(ds, 3) == 3


def test_GridSampler_full_grid(fixture_GridSampler_parabola):
    """Full-grid helper function: special cases.

    These use of `setdiff2d` is to avoid dependency on order of samples.
    """

    ds, ss = fixture_GridSampler_parabola(3)

    assert setdiff2d(ss.full_grid(ds, 1), np.asfarray([[0, 0, 0]])).size == 0

    assert (
        setdiff2d(
            ss.full_grid(ds, 2),
            np.asfarray(
                [
                    [-2, -2, -2],
                    [-2, -2, +2],
                    [-2, +2, -2],
                    [-2, +2, +2],
                    [+2, -2, -2],
                    [+2, -2, +2],
                    [+2, +2, -2],
                    [+2, +2, +2],
                ]
            ),
        ).size
        == 0
    )

    assert (
        setdiff2d(
            ss.full_grid(ds, 3),
            np.asfarray(
                [
                    [-2, -2, -2],
                    [-2, -2, +0],
                    [-2, -2, +2],
                    [-2, +0, -2],
                    [-2, +0, +0],
                    [-2, +0, +2],
                    [-2, +2, -2],
                    [-2, +2, +0],
                    [-2, +2, +2],
                    [+0, -2, -2],
                    [+0, -2, +0],
                    [+0, -2, +2],
                    [+0, +0, -2],
                    [+0, +0, +0],
                    [+0, +0, +2],
                    [+0, +2, -2],
                    [+0, +2, +0],
                    [+0, +2, +2],
                    [+2, -2, -2],
                    [+2, -2, +0],
                    [+2, -2, +2],
                    [+2, +0, -2],
                    [+2, +0, +0],
                    [+2, +0, +2],
                    [+2, +2, -2],
                    [+2, +2, +0],
                    [+2, +2, +2],
                ]
            ),
        ).size
        == 0
    )


def test_GridSampler_sample_validation(fixture_GridSampler_parabola):
    """Sampling of validation set: simple test cases."""

    ds, ss = fixture_GridSampler_parabola(3, 27)

    vs = ss.apply(ds)
    assert setdiff2d(ss.full_grid(ds, 3), vs.samples()).size == 0
    assert vs.samples().shape == (27, 3)
    assert len(vs.labels()) == 27


def test_GridSampler_sample_training(fixture_GridSampler_parabola):
    """Sampling of training set: simple test cases."""

    ds1, ss1 = fixture_GridSampler_parabola(3, 43)
    ds2, ss2 = fixture_GridSampler_parabola(3, 39)

    assert ss1.next_grid_size(ds1, 43) == ss2.next_grid_size(ds2, 39) == 4

    vs1, vs2 = ss1.apply(ds1).samples(), ss2.apply(ds2).samples()
    assert vs1.shape == (43, 3) and vs2.shape == (39, 3)
    assert setdiff2d(vs1, vs2).shape == (4, 3)


def test_GridSampler_sample_deterministic(fixture_GridSampler_parabola):
    """Deterministic reproducability."""

    ds1, ss1 = fixture_GridSampler_parabola(3, 39, rng=1)
    ds2, ss2 = fixture_GridSampler_parabola(3, 39, rng=1)
    vs1 = ss1.apply(ds1).samples()
    vs2 = ss2.apply(ds2).samples()
    assert (vs1 == vs2).all()

    ds3, ss3 = fixture_GridSampler_parabola(3, 39, rng=2)
    vs3 = ss3.apply(ds3).samples()
    assert (vs1 != vs3).any()
