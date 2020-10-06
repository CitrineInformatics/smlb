"""Interface to the Java language.

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
2019-2020, Citrine Informatics.

Some functionality wrapped by smlb is implemented in the Java programming language.
smlb uses the py4j library to interact with Java code.

Code in this file supports other parts of smlb or user code to call Java libraries.
"""

# Notes regarding bridges from Python to Java:
#
# Overview:
# https://talvi.net/a-brief-overview-of-python-java-bridges-in-2020.html
#
# Suitable options include py4j, jpype and pyjnius
# Related: SWIG generates C/C++ bindings to Python, Java and others, but not between Java and Python
#
# py4j was chosen as it is maintained, has conda support, and lolopy uses it.
# While it requires running a Java server, the pre-packaged ones seem sufficient.
# py4j is thread-safe as long as only user-created threads call Java

# Two alternative approaches to using py4j in smlb are
# a) have each using module launch its own Java gateway process
# b) have a single Java gateway process and provide new_jvm_views() to it
# Alternative a) was chosen because b) only allows to set the class path once (there is no portable
# way to change it while the SVM is running). This would effectively require to specify _all_ potentially
# necessary Java includes at the beginning, which is undesirable (strong coupling) and not possible
# for user-provided functionality.


from abc import ABCMeta, abstractmethod
from typing import Optional

import py4j.java_gateway

from smlb import params

# todo: redirect Java output to loggers (requires logging functionality)

# todo: add port number, java options and java path as arguments


class JavaGateway(metaclass=ABCMeta):
    """Provide gateway to Java functionality.

    Uses py4j, see https://www.py4j.org/
    This class should not be included in the top-level smlb __init__.py file
    because py4j is an optional dependency.
    """

    # this implementation uses class members to enable keeping the JVM
    # connection alive across uses. The class members are specific for
    # each _derived_ class.

    _gateway = None  # py4j gateway
    _class_path = ""  # class path argument passed at initialization

    @abstractmethod
    def __init__(self, class_path: Optional[str] = None):
        """Initialize Java gateway.

        If derived class is initialized for the first time,
        start up JVM and create gateway. On subsequent initializations
        of derived class, the same gateway is used, except when a
        different class_path is passed. In that case,
        the JVM is shut down and restarted with the new class path.

        Parameters:
            class_path: local filesystem class path containing one
                or more directories or .jar files. If not specified,
                an empty string is passed as classpath to the JVM.

        Raises:
            BenchmarkError if the class_path is invalid.
        """

        # todo: class_path = params.optional_(class_path, params.string)
        class_path = params.any_(class_path, params.string, params.none)

        if self.__class__._gateway is None:
            # first time derived class is instantiated, create gateway
            self._launch_gateway(class_path=class_path)
        elif self.__class__._class_path != class_path:
            # if parameters changed, restart the JVM
            self._shutdown_gateway()
            self._launch_gateway(class_path=class_path)
        else:
            # subsequent instantiations use the same gateway
            pass

    def _launch_gateway(self, class_path: Optional[str]):
        """ """

        assert self.__class__._gateway is None
        self.__class__._class_path = class_path

        # TODO: explicitly validate that class_path consists of existing directories and filenames.
        #       Passing of non-existing files does NOT yield an error but results in later
        #       references to org.abc.xyz.....Class being JavaPackage, not JavaClass.

        # launch gateway
        self.__class__._gateway = py4j.java_gateway.JavaGateway.launch_gateway(
            classpath=class_path
        )

    def _shutdown_gateway(self):
        """Shut down JVM gatway."""

        assert (
            self.__class__._gateway is not None
        ), "Internal error: attempt to shut down, but no gateway"

        self.__class__._gateway.shutdown()
        self.__class__._gateway = None

    @property
    def gateway(self):
        """Access Java gateway."""

        assert (
            self.__class__._gateway is not None
        ), "Internal error: __init__ failed to launch gateway"

        return self.__class__._gateway
