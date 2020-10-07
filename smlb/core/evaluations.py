"""Evaluations.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2019-2020, Matthias Rupp, Citrine Informatics.

Graphical and textual summaries of results.
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Optional

from smlb import SmlbObject
from smlb import BenchmarkError
from smlb import params


class EvaluationConfiguration(SmlbObject):
    """Base class for evaluation configurations.

    Provided functionality differs by type of evaluation (graphical, textual, other).
    """

    # this class provides a base type for parameter validation purposes

    pass


class Evaluation(SmlbObject, metaclass=ABCMeta):
    """Abstract base class for graphical and textual summaries of results.

    After initialization, an Evaluation is first `evaluate`d and then `render`ed.
    Derived classes should re-define `_render`, not `render`, which takes care of
    initialization and release of resources for the derived class's `_render` method.
    """

    def __init__(self, configuration: Optional[EvaluationConfiguration] = None, **kwargs):
        """Initialize Evaluation.

        Parameters:
            configuration: optional configuration object controlling rendering details
        """

        super().__init__(**kwargs)

        self._configuration = params.any_(
            configuration, lambda arg: params.instance(arg, EvaluationConfiguration), params.none
        )
        if self._configuration is None:
            self._configuration = self._default_configuration()

        self._auxiliary = dict()  # internal handle on optional auxiliary outcome data

    @abstractmethod
    def _default_configuration(self):
        """Query default configuration.

        Returns:
            instance of EvaluationConfiguration or derived class
        """

        raise NotImplementedError

    @property
    def configuration(self):
        """Query configuration.

        Returns:
            configuration object
        """

        return self._configuration

    @property
    def auxiliary(self):
        """Access auxiliary data.

        Empty if no auxiliary data is available (depends on plot type).
        """

        # todo: add a wrapper to return a constant/non-mutable dictionary
        #       to avoid unchecked changes to self._auxiliary

        return self._auxiliary

    def add_auxiliary(self, key: str, value: Any):
        """Add auxiliary information.

        Parameters:
            key: string key for retrieving information later
            value: auxiliary information to store under key

        A setter could have been used, for example, as
        `auxiliary = { key: value }`. This solution was
        considered abuse of notation as the syntax would
        have suggested assignment but would have added instead
        """

        key = params.string(key)

        if self._auxiliary.keys() & key:
            raise BenchmarkError("internal error: non-unique evaluation auxiliary data")

        self._auxiliary[key] = value

    @abstractmethod
    def evaluate(self, results: Any):
        """Evaluate results.

        Evaluation performs any computing necessary for later rendering.
        For example, a LearningCurvePlot would prepare which data to show and compute asymptotic fits.
        Any heavy computation should be performed here.

        Parameters:
            results: benchmark results; type and arrangement of results
                     depend on the specific Evaluation
        """

        pass  # might be called by derived class's evaluate() method

    @abstractmethod
    def render(self):
        """Render evaluation.

        Specific derived classes should override _render(), not this method.
        For example, `Plot` overrides `render`, but `LearningCurvePlot` overrides `_render`.
        """

        raise NotImplementedError

    @abstractmethod
    def _render(self):
        """Render evaluation.

        Specific derived classes should override this method (see `render` method).
        The rendering target will have been initialized when this method is called,
        and will be deconstructed if required after this method returns.
        """

        raise NotImplementedError
