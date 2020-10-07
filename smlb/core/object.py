"""Object.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.

Base class all objects derive from.
"""

from smlb import InvalidParameterError


class SmlbObject:
    """Base class for all smlb objects.

    Provides error handling for initializers.
    """

    def __init__(self, *args, **kwargs):
        """Initialize state.

        Raises InvalidParameterError if any arguments are passed.

        This class's initializer serves as a catch-all for invalid initializer arguments
        in the inheritance chain. Arguments unused by specific initializers are passed
        to base class initializers using super(). Since Object is the last class in
        method resolution order, any arguments passed to it have not been handled by
        any of the classes in the inheritance chain. Such unhandled arguments are
        considered errors.

        Parameters:
            arbitrary arguments and keyword arguments.

        Raises:
            InvalidParameterError if any (keyword) arguments are specified
        """

        if len(args) > 0:
            raise InvalidParameterError(
                "nothing",
                args,
                f"Unhandled positional arguments in '{self.__class__.__name__}.__init__'",
            )
        if len(kwargs) > 0:
            raise InvalidParameterError(
                "nothing",
                kwargs,
                f"Unhandled keyword arguments in '{self.__class__.__name__}'.__init__",
            )
