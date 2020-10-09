"""Parameter validation tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019-2020, Citrine Informatics.
"""

import pytest

import numpy as np

import smlb
from smlb import params, InvalidParameterError

###########
#  basic  #
###########


def test_boolean():
    """Tests for boolean arguments."""

    assert params.boolean(True) is True
    assert params.boolean("true") is True
    assert params.boolean("True") is True

    assert params.boolean(False) is False
    assert params.boolean("false") is False
    assert params.boolean("False") is False

    with pytest.raises(InvalidParameterError):
        params.boolean(0)
    with pytest.raises(InvalidParameterError):
        params.boolean(1.0)


def test_real():
    """Tests for real scalars."""

    assert params.real(1) == 1.0
    assert params.real(-2.0) == -2.0

    for arg in (True, False):
        with pytest.raises(InvalidParameterError):
            params.real(arg)


def test_string():
    """Tests for strings."""

    assert params.string("") == ""
    assert params.string("hello world!") == "hello world!"
    with pytest.raises(InvalidParameterError):
        params.string(1)


def test_callable_1():
    """Test for callables."""

    # lambda function, single argument
    # lambda function argument is position-or-keyword
    f1 = lambda arg: arg + 1  # noqa: E731

    assert params.callable(f1)
    assert params.callable(f1, num_pos_or_kw=1)
    assert params.callable(
        f1, num_pos_only=0, num_pos_or_kw=1, num_var_pos=0, num_kw_only=0, num_var_kw=0
    )
    with pytest.raises(InvalidParameterError):
        assert params.callable(f1, num_pos_or_kw=0)
    with pytest.raises(InvalidParameterError):
        assert params.callable(f1, num_pos_or_kw=-1)

    # regular function, two positional arguments
    def f2(a, b, *args):
        pass

    assert params.callable(f2)
    assert params.callable(f2, num_pos_or_kw=2)
    assert params.callable(f2, num_var_pos=1)

    # function object, only keyword arguments
    class f3c:
        def __call__(self, a=1, b=2, **kwargs):
            pass

    f3 = f3c()

    assert params.callable(f3)
    assert params.callable(
        f3, num_pos_only=0, num_pos_or_kw=2, num_var_pos=0, num_kw_only=0, num_var_kw=1
    )

    # positional-only
    # built-in function len, unlike float, has signature defined in Python 3.6
    assert params.callable(len, num_pos_only=1)

    # keyword-only
    def f4(a, *ignore, b=None, c=1):
        pass

    assert params.callable(f4, num_kw_only=2)

    # mixed case
    def f5(a, b, *c, d=1, e=2, **f):
        pass

    assert params.callable(
        f5, num_pos_or_kw=2, num_pos_only=0, num_var_pos=1, num_kw_only=2, num_var_kw=1
    )


def test_callable_2():
    """Test for non-callables."""

    for arg in (1, 1.0, "s"):
        with pytest.raises(InvalidParameterError):
            params.callable(arg)


###############
#  numerical  #
###############


def test_real_vector_1():
    """Tests real vectors."""

    assert np.array_equal(params.real_vector([1]), np.asfarray([1]))
    assert np.array_equal(params.real_vector([1, 2], dimensions=2), np.asfarray([1, 2]))
    assert np.array_equal(
        params.real_vector([1, 2], dimensions=2, domain=[0, 3]), np.asfarray([1, 2])
    )
    assert np.array_equal(
        params.real_vector([1, 2], domain=[[0.5, 1.5], [0, 3]]), np.asfarray([1, 2])
    )

    with pytest.raises(InvalidParameterError):
        params.real_vector([1, 2], dimensions=3)
    with pytest.raises(InvalidParameterError):
        params.real_vector([1, 2], domain=[0, 1.5])


def test_real_vector2():
    """Accumulated test cases."""

    # only (a,b) is extended, [(a,b)] is not
    assert (
        params.real_vector([0.5, 0.5, 1], dimensions=3, domain=(0, np.inf))
        == np.asfarray([0.5, 0.5, 1])
    ).all()
    with pytest.raises(InvalidParameterError):
        params.real_vector([0.5, 0.5, 1], dimensions=3, domain=[(0, np.inf)])


def test_hypercube_domain_1():
    """Tests hypercube vector space domains."""

    # without dimensionality
    assert (params.hypercube_domain([[1, 2]]) == np.asfarray([[1, 2]])).all()
    assert (params.hypercube_domain([[1, 2], [3, 4]]) == np.asfarray([[1, 2], [3, 4]])).all()
    with pytest.raises(InvalidParameterError):
        params.hypercube_domain([1, 2])
    with pytest.raises(InvalidParameterError):
        params.hypercube_domain([[2, 1]])

    # with dimensionality
    assert (params.hypercube_domain([[1, 2]], dimensions=1) == np.asfarray([[1, 2]])).all()
    assert (
        params.hypercube_domain([[1, 2], [3, 4]], dimensions=2) == np.asfarray([[1, 2], [3, 4]])
    ).all()
    assert (
        params.hypercube_domain([1, 2], dimensions=3) == np.asfarray([[1, 2], [1, 2], [1, 2]])
    ).all()
    with pytest.raises(InvalidParameterError):
        params.hypercube_domain([[1, 2], [2, 1]], dimensions=2)


def test_hypercube_domain_2():
    """Accumulated test cases."""

    assert (params.hypercube_domain([(0, np.inf)]) == np.asfarray([[0, np.inf]])).all()
    assert (params.hypercube_domain((0, np.inf), dimensions=1) == np.asfarray([[0, np.inf]])).all()


def test_interval():
    """Tests real intervals."""

    assert params.interval((1, 2)) == (1.0, 2.0)
    assert params.interval((1, 2), from_=1, to=2) == (1.0, 2.0)
    assert params.interval((1, 22), above=1.9999999) == (1.0, 22.0)
    assert params.interval((1, 2), below=2.0000001) == (1.0, 2.0)
    assert params.interval((-3, -1), from_=-3, below=-0.9999999) == (-3.0, -1.0)

    for arg in (None, 1, 1.0, "a", (1, 2, 3), (1,)):
        with pytest.raises(InvalidParameterError):
            params.interval(arg)
    with pytest.raises(InvalidParameterError):
        params.interval((1.1, 1.0))


######################
#  chemical_element  #
######################


def test_chemical_element():
    """Tests chemical element validation."""

    assert params.chemical_element(6) == 6
    assert params.chemical_element("C") == 6

    assert [params.chemical_element(i) for i in range(1, 119)] == list(range(1, 119))
    abrvs = [smlb.element_data(i, "abbreviation") for i in range(1, 119)]
    assert [params.chemical_element(z) for z in abrvs] == list(range(1, 119))


##########
#  meta  #
##########


def test_optional_():
    """Test optional_ meta test."""

    # only testf and None are valid
    assert params.optional_(None, params.integer) is None
    assert params.optional_(1, params.integer) == 1
    with pytest.raises(InvalidParameterError):
        params.optional_("x", params.integer)
    with pytest.raises(InvalidParameterError):
        params.optional_(1, lambda arg: params.integer(arg, above=1))

    # default value
    assert params.optional_(1, params.integer, default=2) == 1
    assert params.optional_(None, params.integer, default=2) == 2


def test_any_():
    """Tests any_ meta test."""

    # special case: single test
    assert params.any_(None, lambda arg: params.none(arg)) is None
    with pytest.raises(InvalidParameterError):
        params.any_("_", lambda arg: params.none(arg))

    # special case: or
    assert params.any_(None, lambda arg: params.none(arg), lambda arg: params.none(arg)) is None
    assert params.any_(None, lambda arg: params.none("_"), lambda arg: params.none(arg)) is None
    assert params.any_(None, lambda arg: params.none(arg), lambda arg: params.none("_")) is None
    with pytest.raises(InvalidParameterError):
        params.any_(None, lambda arg: params.none("_"), lambda arg: params.none("_"))

    # three tests
    assert (
        params.any_(
            None,
            lambda arg: params.none(arg),
            lambda arg: params.none(arg),
            lambda arg: params.none(arg),
        )
        is None
    )
    assert (
        params.any_(
            None,
            lambda arg: params.none(arg),
            lambda arg: params.none(arg),
            lambda arg: params.none("_"),
        )
        is None
    )
    assert (
        params.any_(
            None,
            lambda arg: params.none(arg),
            lambda arg: params.none("_"),
            lambda arg: params.none(arg),
        )
        is None
    )
    assert (
        params.any_(
            None,
            lambda arg: params.none(arg),
            lambda arg: params.none("_"),
            lambda arg: params.none("_"),
        )
        is None
    )
    assert (
        params.any_(
            None,
            lambda arg: params.none("_"),
            lambda arg: params.none(arg),
            lambda arg: params.none(arg),
        )
        is None
    )
    assert (
        params.any_(
            None,
            lambda arg: params.none("_"),
            lambda arg: params.none(arg),
            lambda arg: params.none("_"),
        )
        is None
    )
    assert (
        params.any_(
            None,
            lambda arg: params.none("_"),
            lambda arg: params.none("_"),
            lambda arg: params.none(arg),
        )
        is None
    )
    with pytest.raises(InvalidParameterError):
        params.any_(
            None,
            lambda arg: params.none("_"),
            lambda arg: params.none("_"),
            lambda arg: params.none("_"),
        )


def test_all_():
    """Tests all_ meta test."""

    # special case: single test
    assert params.any_(None, lambda arg: params.none(arg)) is None
    with pytest.raises(InvalidParameterError):
        params.all_("_", lambda arg: params.none(arg))

    # special case: and
    assert (
        params.all_(
            2, lambda arg: params.integer(arg, above=1), lambda arg: params.integer(arg, from_=2)
        )
        == 2
    )
    assert (
        params.all_(
            3,
            lambda arg: params.integer(arg, above=1),
            lambda arg: params.integer(arg, from_=2),
            lambda arg: params.integer(arg, from_=3),
        )
        == 3
    )

    # fail in first testf
    with pytest.raises(InvalidParameterError):
        params.all_(
            1, lambda arg: params.integer(arg, above=1), lambda arg: params.integer(arg, from_=2)
        )

    # fail in last testf
    with pytest.raises(InvalidParameterError):
        params.all_(
            2,
            lambda arg: params.integer(arg, above=1),
            lambda arg: params.integer(arg, from_=2),
            lambda arg: params.integer(arg, above=2),
        )


def test_tuple_():
    """Tests tuple_ meta test."""

    testf = lambda arg: params.none(arg)

    # special case: no tuple
    with pytest.raises(InvalidParameterError):
        params.tuple_(None, lambda arg: arg)

    # special case: single test
    assert params.tuple_((None,), testf) == (None,)
    with pytest.raises(InvalidParameterError):
        params.any_("_", testf)

    # special case: 2-tuple
    assert params.tuple_((None, None), testf, testf) == (None, None)
    with pytest.raises(InvalidParameterError):
        params.tuple_(("_", None), testf, testf)
    with pytest.raises(InvalidParameterError):
        params.tuple_((None, "_"), testf, testf)
    with pytest.raises(InvalidParameterError):
        params.tuple_(("_", "_"), testf, testf)

    # arity parameter
    assert params.tuple_((None, None), testf, arity=2)
    with pytest.raises(InvalidParameterError):
        params.tuple_((None, None), testf, arity=3)
    with pytest.raises(InvalidParameterError):
        params.tuple_((None, None, None), testf, arity=2)

    # default parameter
    assert params.tuple_((None,), testf, arity=3, default=None) == (None, None, None)

    # no arity, no default
    assert params.tuple_((None, None, None), testf) == (None, None, None)
