"""Utilities.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.

Auxiliary code.
"""

import numpy as np

from smlb import InvalidParameterError


def is_sequence(arg):
    """True if argument is a list, tuple, array or similar object, but not a string, dictionary, set or similar object.

    Parameters:
        arg: the object to test

    Returns:
        True or False
    """

    # np.float{16,32,64} and np.int types have __getitem__ defined
    # this is a long-standing bug in NumPy and unlikely to be fixed
    # todo: backport to qmmlpack, write tests
    if isinstance(arg, (str, bytes, np.number, dict, set)):
        return False

    return hasattr(arg, "__getitem__") or hasattr(arg, "__iter__")


def which(*args):
    """'which' statement.

    which(
        cond_1, value_1,
        cond_2, value_2,
        ...
    )

    Returns value_i for the first condition cond_i that is true.
    It is an error if none of the cond_i are true.

    For default values, use 'which(cond_1, value_1, ..., True, default)',
    where the 'True' can be omitted, that is, a last default value can be specified:
    which(cond_1, value_1, ..., cond_k, value_k, default)
    """

    if len(args) == 0:
        raise InvalidParameterError(
            "conditions and cases", "nothing", explanation="'which' statement without arguments"
        )
    if len(args) % 2 == 1:
        return which(*args[:-1], True, args[-1])
    for i in range(0, len(args), 2):
        if args[i]:
            return args[i + 1]
    raise InvalidParameterError(
        "at least one condition applies",
        "no condition applied",
        explanation="'which' command fell through (no case applied)",
    )
