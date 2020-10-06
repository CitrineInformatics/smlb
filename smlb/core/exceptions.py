"""Exceptions.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.

Error handling. 

All benchmark exceptions derive from BenchmarkError.
"""


class BenchmarkError(Exception):
    """Signals failure in benchmarking.

    This is the exception generally raised within the benchmark.
    All more specific exceptions derive from it."""

    pass


class InvalidParameterError(BenchmarkError):
    """Raised for invalid parameters.

    Parameters of functions and methods can be invalid due to
    invalid values (for example, out of range) and due to
    invalid types (for example, string instead of number).
    """

    def __init__(self, expected, got, *args, explanation=None, **kwargs):
        """Initialize exception.

        Parameters:
            expected: textual description of expected parameter
            got: actual parameter passed
            explanation: if specified, additional explanation for the error;
                         the explanation will be appended to the error message

        Any other arguments are passed to the parent class.
        """

        self.expected, self.got, self.explanation = expected, got, explanation

        message = f"Expected {expected}, got '{got}'"
        if explanation is not None:
            message += "\n" + explanation

        super(InvalidParameterError, self).__init__(message, *args, **kwargs)


# InvalidParameterValue will likely not be pickleable in its current form. See here for a fix if it becomes necessary:
# https://stackoverflow.com/questions/16244923/how-to-make-a-custom-exception-class-with-multiple-init-args-pickleable
