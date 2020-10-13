"""Parameters.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.

Utility functions to validate and canonicalize parameters.
"""

from typing import Optional, Union

import numpy as np

from smlb import InvalidParameterError
from smlb import is_sequence, which

# Circular import dependencies
#
# 'distributions.py' imports 'parameters.py' (this file),
# but this file also requires classes from 'distributions.py'.
# One solution is to import 'distributions.py' at end of file.
# See http://effbot.org/zone/import-confusion.htm
# Another solution, employed here, is to import 'distributions.py'
# within the functions where its functionality is used.

# todo: write more unit tests


class params:
    """Validate and canonicalize parameters.

    Parameters are tested for validity and are brought into
    a canonical form. The validated canonicalized parameter
    is returned. In case of errors, InvalidParameterError is raised.

    Recommended usage example:

        def f(n):
            n = params.integer(n)

    Note that this is unproblematic even for mutable parameters as
    the parameter is re-assigned.
    """

    ###########
    #  Basic  #
    ###########

    @staticmethod
    def none(arg):
        """None.

        Parameters:
            arg: parameter to validate as None

        Returns:
            None

        Raises:
            InvalidParameterError: if arg is invalid
        """

        ipe = InvalidParameterError("None", arg)

        if arg is not None:
            raise ipe

        return None

    @staticmethod
    def boolean(arg):
        """True or False.

        Accepts values True, False, "true", "True", "false", "False".
        Rejects values 0, 1, 0., 1.

        Parameters:
            arg: parameter to validate as boolean

        Returns:
            built-in boolean

        Raises:
            InvalidParameterError: if arg is invalid

        Acceptance of all objects `arg` for which `bool(arg)` works
        would lead to subtle bugs, for example, when testing for bool or float,
        `params.any_(arg, lambda arg: params.boolean(arg), lambda arg: params.real(arg))`
        would yield True due to conversion of real to bool.
        """

        ipe = InvalidParameterError("boolean", arg)

        # try: arg = bool(arg) as test would lead to bugs, see docstring.
        # arg in {False, "false", "False"} fails as 0 == False
        if arg is True or arg in {"true", "True"}:
            arg = True
        elif arg is False or arg in {"false", "False"}:
            arg = False
        else:
            raise ipe

        return arg

    @staticmethod
    def integer(arg, from_=None, to=None, above=None, below=None):
        """Integer number.

        Negative, non-negative, positive, non-positive integers are special cases.

        Parameters:
            arg: parameter to validate as an integer
            from_: if specified, lowest admissible number (closed set lower bound)
            to: if specified, highest admissible number (closed set upper bound)
            above: if specified, highest non-admissible number (open set lower bound)
            below: if specified, lowest non-admissible number (open set upper bound)

        Returns:
            built-in integer type

        Raises:
            InvalidParameterError: if parameter arg is invalid
        """

        # TODO: throw exception on non-integer floating point values
        bounded = not (from_ is None and to is None and above is None and below is None)
        msg = f"{'bounded ' if bounded else ''}integer"
        if bounded:
            msg += (
                "("
                + ("" if from_ is None else f"from {from_}, ")
                + ("" if above is None else f"above {above}, ")
                + ("" if to is None else f"to {to}, ")
                + ("" if below is None else f"below {below}, ")
            )
            msg = msg[:-2] + ")"
        ipe = InvalidParameterError(msg, arg)

        try:
            arg = int(arg)

            if from_ is not None:
                if arg < from_:
                    raise ipe
            if to is not None:
                if arg > to:
                    raise ipe
            if above is not None:
                if arg <= above:
                    raise ipe
            if below is not None:
                if arg >= below:
                    raise ipe
        except Exception as e:
            raise ipe from e

        return arg

    @staticmethod
    def real(arg, from_=None, to=None, above=None, below=None):
        """Real number, floating point type.

        Parameters:
            arg: parameter to validate as a real numbre
            from_: if specified, lowest admissible number (closed set lower bound)
            to: if specified, highest admissible number (closed set upper bound)
            above: if specified, highest non-admissible number (open set lower bound)
            below: if specified, lowest non-admissible number (open set upper bound)

        Returns:
            built-in floating point type

        Raises:
            InvalidParameterError: for invalid parameter arg
        """

        bounded = not (from_ is None and to is None and above is None and below is None)
        ipe = InvalidParameterError(f"{'bounded ' if bounded else ''}real number", arg)

        try:
            # guard against True and False, which are convertible to float
            if arg is True or arg is False:
                raise ipe

            arg = float(arg)

            if from_ is not None:
                if arg < from_:
                    raise ipe
            if to is not None:
                if arg > to:
                    raise ipe
            if above is not None:
                if arg <= above:
                    raise ipe
            if below is not None:
                if arg >= below:
                    raise ipe
        except Exception as e:
            raise ipe from e

        return arg

    @staticmethod
    def string(arg):
        """A string.

        The argument must be a string, no conversion from non-string types.

        Currently supports only built-in string type.

        Parameters:
            arg: parameter to validate as string

        Returns:
            arg as string

        Raises:
            InvalidParameterError if arg is not a string
        """

        ipe = InvalidParameterError("string", arg)

        try:
            # str(arg) would convert from non-string types
            return params.instance(arg, str)  # only str is currently supported
        except Exception as e:
            raise ipe from e

    @staticmethod
    def enumeration(arg, values):
        """One of a finite number of choices (enumerated or categorical variable).

        Parameters:
            arg: parameter to validate
            values: set of all valid values for data

        Returns:
            one of the members of `values`

        Raises:
            InvalidParameterError: if arg is invalid
        """

        ipe = InvalidParameterError(f"categorical variable", arg)

        try:
            # arg in values can fail if arg is not hashable
            if arg not in values:
                raise ipe
        except Exception as e:
            raise ipe from e

        return arg

    @staticmethod
    def sequence(arg, length=None, type_=None, testf=None):
        """Sequence.

        Sequence, of given length and type if specified.

        Parameters:
            arg: parameter to be validated as a sequence
            length: required length of sequence or None (default)
            type_: required type for all sequence elements or None (default)

        Returns:
            arg if a sequence

        Raises:
            InvalidParameterError if arg is not a sequence, of given length and type if specified
        """

        ipe_length = "" if length is None else f" of length {length}"
        ipe_type = "" if type_ is None else f" of type {type(type_).__name__}"
        ipe_testf = "" if testf is None else " with constraints"
        ipe = InvalidParameterError(f"a sequence{ipe_length}{ipe_type}{ipe_testf}", arg)

        if not is_sequence(arg):
            raise ipe
        if length is not None:
            if len(arg) != length:
                raise ipe
        if type_ is not None:
            if not all(isinstance(el, type_) for el in arg):
                raise ipe
        if testf is not None:
            try:
                for el in arg:
                    testf(el)
            except Exception as e:
                raise ipe from e

        return arg

    @staticmethod
    def instance(arg, class_):
        """Instance of a given class.

        Parameters:
            arg: parameter to be validated as an instance of a class
            class_: arg must be derived from class_ or a subclass of it

        Returns:
            object
        """

        ipe = InvalidParameterError(f"instance of '{class_.__name__}'", type(arg).__name__)

        if not isinstance(arg, class_):
            raise ipe

        return arg

    @staticmethod
    def callable(
        arg,
        num_pos_only: Optional[int] = None,
        num_pos_or_kw: Optional[int] = None,
        num_var_pos: Optional[int] = None,
        num_kw_only: Optional[int] = None,
        num_var_kw: Optional[int] = None,
    ):
        """Callable entity such as a function or object with __call__.

        Most functions will have positional_or_keyword, var_positional or var_keyword parameters.

        Example:
            `def f5(a, b, *c, d=1, e=2, **f): pass`
            has 2 positional-or-keyword, 1 variable-positional, 2 keyword-only and 1 variable-keyword arguments.

        Positional-only parameters refer to use of '/', which is a notational convention mostly
        used for built-in functions until Python 3.7, and available as a keyword from Python 3.8 on.

        Parameters:
            arg: parameter to be validated as callable
            num_pos_only: if specified, callable must have this number of position-only arguments
            num_pos_or_kw: if specified, callable must have this number of positional-or-keyword arguments
            num_var_pos: if specified, callable must have this number of variable-positional ('*args') arguments
            num_kw_only: if specified, callable must have this number of keyword-only arguments
            num_var_kw: if specified, callable must have this number of variable-keyword ('**kwargs') arguments

        Returns:
            callable as passed

        Raises:
            InvalidParameterError if arg is not a callable or a specified number of parameters is not met
        """

        nneg_int_f = lambda arg: params.integer(arg, from_=0)
        num_pos_only = params.optional_(num_pos_only, nneg_int_f)
        num_pos_or_kw = params.optional_(num_pos_or_kw, nneg_int_f)
        num_var_pos = params.optional_(num_var_pos, nneg_int_f)
        num_kw_only = params.optional_(num_kw_only, nneg_int_f)
        num_var_kw = params.optional_(num_var_kw, nneg_int_f)

        import inspect  # lazy import

        ipe = InvalidParameterError("callable object", type(arg).__name__)

        try:
            s = inspect.signature(arg)  # raises if not callable

            if not (
                num_pos_only is None
                and num_pos_or_kw is None
                and num_var_pos is None
                and num_kw_only is None
                and num_var_kw is None
            ):
                arg_kinds = tuple(arg.kind for arg in s.parameters.values())
                pc = inspect.Parameter  # shortcut for inspect.Parameter class
                n_pos_only, n_pos_or_kw, n_var_pos, n_kw_only, n_var_kw = tuple(
                    len(tuple(kind for kind in arg_kinds if kind == kkind))
                    for kkind in (
                        pc.POSITIONAL_ONLY,
                        pc.POSITIONAL_OR_KEYWORD,
                        pc.VAR_POSITIONAL,
                        pc.KEYWORD_ONLY,
                        pc.VAR_KEYWORD,
                    )
                )

                if num_pos_only is not None:
                    if n_pos_only != num_pos_only:
                        raise InvalidParameterError(
                            f"callable object ({num_pos_only} positional-only parameters)",
                            f"{type(arg).__name__} with {n_pos_only} positional-only parameters",
                        )
                if num_pos_or_kw is not None:
                    if n_pos_or_kw != num_pos_or_kw:
                        raise InvalidParameterError(
                            f"callable object ({num_pos_or_kw} positional-or-keyword parameters)",
                            f"{type(arg).__name__} with {n_pos_or_kw} positional-or-keyword parameters",
                        )

                if num_var_pos is not None:
                    if n_var_pos != num_var_pos:
                        raise InvalidParameterError(
                            f"callable object ({num_var_pos} positional-variational parameters)",
                            f"{type(arg).__name__} with {n_var_pos} positional-variational parameters",
                        )

                if num_kw_only is not None:
                    if n_kw_only != num_kw_only:
                        raise InvalidParameterError(
                            f"callable object ({num_kw_only} keyword-only parameters)",
                            f"{type(arg).__name__} with {n_kw_only} keyword-only parameters",
                        )

                if num_var_kw is not None:
                    if n_var_kw != num_var_kw:
                        raise InvalidParameterError(
                            f"callable object ({num_var_kw} keyword-variational parameters)",
                            f"{type(arg).__name__} with {n_var_kw} keyword-variational parameters",
                        )

        except (ValueError, TypeError) as e:
            raise ipe from e

        return arg

    ###############
    #  Numerical  #
    ###############

    @staticmethod
    def numpy_array(arg, dtype=None):
        """Any NumPy array.

        Tests if argument is a NumPy array, of any dtype (if not specified), of any dimensionality.

        Parameters:
            arg: parameter to validate
            dtype: dtype of arg if not None (default)

        Returns:
            NumPy array, of given dtype if specified

        Raises:
            InvalidParameterError: for invalid parameters
        """

        ipe = InvalidParameterError(
            f"NumPy array{'' if dtype is None else 'of dtype ' + str(dtype)}", arg
        )

        try:
            arg = np.asarray(arg) if dtype is None else np.asarray(arg, dtype=dtype)
        except Exception as e:
            raise ipe from e

        return arg

    @staticmethod
    def numpy_rectangular_array(arg, dtype=None):
        """Rectangular NumPy array.

        Parameters:
            arg: parameter to validate as rectangular NumPy array
            dtype: dtype of arg if not None (default)

        Returns:
            rectangular NumPy array with given dtype if specified

        Raises:
            InvalidParameterError: for invalid parameters
        """

        ipe = InvalidParameterError(
            f"rectangular NumPy array{'' if dtype is None else 'of dtype ' + str(dtype)}", arg
        )

        arg = params.numpy_array(arg, dtype=dtype)
        if len(arg.shape) != 2:
            raise ipe

        return arg

    @staticmethod
    def real_vector(arg, dimensions: Optional[int] = None, domain=None):
        """Element of a real vector space.

        Parameters:
            arg: parameter to validate
            dimensions: dimensionality of the vector; None (default) if arbitrary
            domain: hypercube domain or None (default)

        Returns:
            vector as NumPy floating point array

        Raises:
            InvalidParameterError: if arg is invalid

        Example:
            params.real_vector(v, dimensions=3, domain=[[0,1], [0,1], [-0.5,0.5]]
        """

        dimensions = params.any_(dimensions, lambda arg: params.integer(arg, above=0), params.none)

        s = f"{'' if domain is None else 'domain-bound '}real vector of dimension '{dimensions}'"
        ipe = InvalidParameterError(s, arg)

        try:
            res = np.asfarray(arg)
            if len(res.shape) != 1:
                raise ipe

            if dimensions is not None:
                if res.shape != (dimensions,):
                    raise ipe

            if domain is not None:
                domain = params.hypercube_domain(domain, dimensions=len(res))
                if (domain[:, 0] > res).any() or (res > domain[:, 1]).any():
                    raise ipe
        except Exception as e:
            raise ipe from e

        return res

    @staticmethod
    def real_matrix(arg, nrows=None, ncols=None):
        """Element of a two-dimensional real vector space.

        Two-dimensional NumPy matrix with `float` dtype.

        Parameters:
            arg: parameter to be validated as a real matrix
            nrows: required number of rows if specified
            ncols: required number of columns if specified

        Returns:
            NumPy 2d array, with given number of rows and columns if specified
        """

        ipe_dims = which(
            nrows is None and ncols is None,
            "",
            nrows is None and ncols is not None,
            f" with {ncols} columns",
            nrows is not None and ncols is None,
            f" with {nrows} rows",
            nrows is not None and ncols is not None,
            f" with {nrows} rows and {ncols} columns",
        )
        ipe = InvalidParameterError(f"real matrix{ipe_dims}", arg)

        try:
            res = np.asfarray(arg)
            if res.shape == (0,):
                res = np.empty((0, 0), dtype=float)  # special case of empty matrix
            if len(res.shape) != 2:
                raise ipe

            if nrows is not None and res.shape[0] != nrows:
                raise ipe
            if ncols is not None and res.shape[1] != ncols:
                raise ipe
        except Exception as e:
            raise ipe from e

        return res

    @staticmethod
    def hypercube_domain(arg, dimensions: Optional[int] = None):
        """A hypercube domain in a real vector space.

        A sequence of ranges [a,b].

        If only a single interval is passed as `arg`,
        it is extended to match the dimensionality.
        For this, dimensionality needs to be specified.

        If dimensionality is not specified, any sequence
        of ranges is accepted. In this case, a single range
        is not accepted, as it can not be reliably extended.

        Parameters:
            arg: sequence of ranges [a,b]
            dimensions: dimensionality of sequence

        Returns:
            2d NumPy array of shape (dimensions,2) and dtype float
        """

        # if empty vector spaces are needed, change to from_=0 to allow zero dimensions
        dimensions = params.any_(dimensions, lambda arg: params.integer(arg, from_=1), params.none)

        ipe = InvalidParameterError(
            f"{'d' if dimensions is None else dimensions}-dimensional hypercube domain", arg
        )

        try:
            res = np.asarray(arg, dtype=float)
            if len(res.shape) not in (1, 2):
                raise ipe

            if len(res.shape) == 1:
                if dimensions is None:
                    raise ipe
                else:
                    res = np.tile(res, (dimensions, 1))

            if dimensions is not None and res.shape != (dimensions, 2):
                raise ipe

            if (res[:, 0] > res[:, 1]).any():
                raise ipe
        except Exception as e:
            raise ipe from e

        return res

    @staticmethod
    def interval(arg, from_=None, to=None, above=None, below=None):
        """Intervals [a,b], [a,b), (a,b], (a,b), 2-tuple of floats.

        If bounds are not specified, interval can be anywhere on the real line.
        It is always required that a <= b.

        Parameters:
            arg: parameter to validate as an interval
            from_: if specified, lowest admissible number (closed lower bound)
            to: if specified, highest admissible number (closed upper bound)
            above: if specified, highest non-admissible number (open lower bound)
            below: if specified, lowest non-admissible number (open upper bound)

        Returns:
            tuple (a,b) of two floating point numbers a and b with a <= b

        Raises:
            InvalidParameterError: for invalid parameter arg
        """

        ipe = InvalidParameterError("interval", arg)

        try:
            arg = params.tuple_(
                arg,
                lambda a: params.real(a, from_=from_, below=below),
                lambda b: params.real(b, to=to, above=above),
                arity=2,
            )

            if arg[0] > arg[1]:
                raise ipe
        except Exception as e:
            raise ipe from e

        return arg

    #################
    #  Statistical  #
    #################

    @staticmethod
    def distribution(arg):
        """Predictive distribution.

        Parameters:
            arg: parameter to validate; predictive distributions;
                 a sequence is interpreted as specifying the means of a DeltaPredictiveDistribution

        Returns:
            PredictiveDistribution or subclass

        Raises:
            InvalidParameterError: if arg is invalid
        """

        # due to circular dependency
        from .distributions import PredictiveDistribution, DeltaPredictiveDistribution

        ipe = InvalidParameterError("distribution", arg)

        try:
            if isinstance(arg, PredictiveDistribution):
                pass
            elif is_sequence(arg):
                # interpret as sequence of means
                arg = np.asfarray(arg)
                if len(arg.shape) != 1:
                    raise ipe
                arg = DeltaPredictiveDistribution(arg)
            else:
                raise ipe
        except Exception as e:
            raise ipe from e

        return arg

    @staticmethod
    def normal_distribution(arg):
        """Predictive normal distribution.

        Parameters:
            arg: parameter to validate; normal predictive distributions;
                 a pair of two same-length sequences is interpreted as
                 means and standard deviations of independent normal predictive distributions

        Returns:
            NormalPredictiveDistribution

        Raises:
            InvalidParameterError: if arg is invalid
        """

        # due to circular dependency
        from .distributions import NormalPredictiveDistribution

        ipe = InvalidParameterError("normal distribution", arg)

        try:
            if isinstance(arg, NormalPredictiveDistribution):
                pass
            elif (
                is_sequence(arg)
                and len(arg) == 2
                and is_sequence(arg[0])
                and is_sequence(arg[1])
                and len(arg[0]) == len(arg[1])
            ):
                # interpret as pair of two same-length sequences
                arg = NormalPredictiveDistribution(arg[0], arg[1])
            else:
                raise ipe  # check if arg is a normal distribution
        except Exception as e:
            raise ipe from e

        return arg

    ################
    #  Scientific  #
    ################

    # chemical elements helper data structure
    # maps textual abbreviation of an element to its atomic number
    _element_atomic_number = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
        "Sc": 21,
        "Ti": 22,
        "V": 23,
        "Cr": 24,
        "Mn": 25,
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
        "Ga": 31,
        "Ge": 32,
        "As": 33,
        "Se": 34,
        "Br": 35,
        "Kr": 36,
        "Rb": 37,
        "Sr": 38,
        "Y": 39,
        "Zr": 40,
        "Nb": 41,
        "Mo": 42,
        "Tc": 43,
        "Ru": 44,
        "Rh": 45,
        "Pd": 46,
        "Ag": 47,
        "Cd": 48,
        "In": 49,
        "Sn": 50,
        "Sb": 51,
        "Te": 52,
        "I": 53,
        "Xe": 54,
        "Cs": 55,
        "Ba": 56,
        "La": 57,
        "Ce": 58,
        "Pr": 59,
        "Nd": 60,
        "Pm": 61,
        "Sm": 62,
        "Eu": 63,
        "Gd": 64,
        "Tb": 65,
        "Dy": 66,
        "Ho": 67,
        "Er": 68,
        "Tm": 69,
        "Yb": 70,
        "Lu": 71,
        "Hf": 72,
        "Ta": 73,
        "W": 74,
        "Re": 75,
        "Os": 76,
        "Ir": 77,
        "Pt": 78,
        "Au": 79,
        "Hg": 80,
        "Tl": 81,
        "Pb": 82,
        "Bi": 83,
        "Po": 84,
        "At": 85,
        "Rn": 86,
        "Fr": 87,
        "Ra": 88,
        "Ac": 89,
        "Th": 90,
        "Pa": 91,
        "U": 92,
        "Np": 93,
        "Pu": 94,
        "Am": 95,
        "Cm": 96,
        "Bk": 97,
        "Cf": 98,
        "Es": 99,
        "Fm": 100,
        "Md": 101,
        "No": 102,
        "Lr": 103,
        "Rf": 104,
        "Db": 105,
        "Sg": 106,
        "Bh": 107,
        "Hs": 108,
        "Mt": 109,
        "Ds": 110,
        "Rg": 111,
        "Cn": 112,
        "Nh": 113,
        "Fl": 114,
        "Mc": 115,
        "Lv": 116,
        "Ts": 117,
        "Og": 118,
    }

    @staticmethod
    def chemical_element(arg: Union[int, str]) -> int:
        """Chemical element.

        Can be specified either via proton number (int) or abbreviation (str).

        Parameters:
            arg: parameter to be validated as a chemical element specification

        Returns:
            proton number corresponding to element

        Raises:
            InvalidParameterError: for invalid parameters
        """

        ipe = InvalidParameterError("chemical element", arg)

        if isinstance(arg, str):
            if arg not in params._element_atomic_number:
                raise ipe
            arg = params._element_atomic_number[arg]

        if not (int(arg) and 1 <= arg <= 118):  # convertible to int and in range
            raise ipe

        return arg

    #  ##########
    #  #  Meta  #
    #  ##########

    @staticmethod
    def optional_(arg, testf, default=None):
        """Optional argument, can be None or something else.

        This is a shorthand for `params.any_(arg, testf, params.none)`, but more explicit.

        Parameters:
            arg: parameter to validate
            testf: test function that accepts a single argument and validates it
            default: if arg is None, the specified default is returned

        Returns:
            either default (if arg is None) or the result of testf(arg)

        Raises:
            InvalidParameterError if testf(arg) raises InvalidParameterError
        """

        ipe = InvalidParameterError("None or successful test", "not None and test failed")

        if arg is None:
            return default

        try:
            return testf(arg)
        except InvalidParameterError as e:
            raise ipe from e

    @staticmethod
    def any_(arg, testf, *args):
        """Logical or/union meta-test.

        At least one of several tests is valid.
        Logical or is a special case of any_.

        Parameters:
            arg: parameter to validate
            testf: test function that accepts a single argument and validates it
            arbitrarily many further functions can be passed

        Returns:
            if only one test function is passed, result of testf(arg) if successful;
            otherwise, result of any(arg, *args)

        Raises:
            InvalidParameterError if testf is the only function passed and testf(arg) raises InvalidParameterError
        """

        ipe = InvalidParameterError("at least one test successful (any_)", "all tests failed")

        try:
            return testf(arg)
        except InvalidParameterError:
            if len(args) == 0:
                raise ipe
            return params.any_(arg, *args)

    @staticmethod
    def all_(arg, testf, *args):
        """Logical and/intersection meta-test.

        All of several tests are valid.
        Logical and is a special case of all_.

        Parameters:
            arg: parameter to validate
            testf: function accepting a single argument
            arbitrarily many further functions can be passed

        Returns:
            if all test functions pass, arg if successful, raises otherwise

        Raises:
            InvalidParameterError if any testf raises an InvalidParameterError
        """

        ipe = InvalidParameterError("all tests successful (all_)", "a test failed")

        try:
            testf(arg)
            if len(args) > 0:
                params.all_(arg, *args)
            return arg
        except InvalidParameterError as e:
            raise ipe from e

    # unique object to allow None as valid parameter for default
    NONE = "49djrh42bvd0Wjd83jfn3D4fjfj4nmfkDFD8Pfj4n4ndf83j"

    @staticmethod
    def tuple_(arg, testf, *args, arity=None, default=NONE):
        """k-tuple meta-test.

        If arity is larger than the number of test functions provided,
        the last test function is repeatedly used. This enables
        `tuple(..., f, arity=3)` for homogeneous-type tuples.

        Parameters:
            arg: parameter to validate as a tuple
            testf: test function that accepts a single argument and validates it
            arbitrarily many further test functions can be passed
            arity: length of tuple
            default: if specified and arity as well, too-short tuples are extended with default value

        Returns:
            arg if it is a tuple and every component is successfully validated

        Raises:
            InvalidParameterError if arg is not a sequence or one of the test functions fails
        """

        if arity is None:
            arity = max(len(args) + 1, len(arg) if is_sequence(arg) else 0)
        ipe = InvalidParameterError(f"{arity}-tuple with valid components (tuple_)", arg)

        if not is_sequence(arg) or len(arg) > arity:
            raise ipe

        if len(arg) < arity:
            if default != params.NONE:
                arg = arg + tuple(default for _ in range(arity - len(arg)))
            else:
                raise ipe

        try:
            testf = (testf, *args)
            return tuple(testf[i if i < len(testf) else -1](arg[i]) for i in range(arity))
        except InvalidParameterError as e:
            raise ipe from e
