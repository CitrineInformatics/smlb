"""Workflows.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2020, Matthias Rupp, Citrine Informatics.

Workflows are instruction sets for benchmarks.
"""

from abc import ABCMeta, abstractmethod

from .object import SmlbObject

# todo: Workflow should provide interface and routines to
#       handle execution and caching of task results


class Workflow(SmlbObject, metaclass=ABCMeta):
    """Instruction set for benchmarks.

    All necessary information is provided at initialization.
    Running a workflow executes its tasks in randomized order.
    """

    @abstractmethod
    def run(self):
        """Executes workflow."""

        raise NotImplementedError
