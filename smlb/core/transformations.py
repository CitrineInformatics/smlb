"""Data transformations.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.

DataTransformations are functors (function objects) that can be fitted 
to data and then applied to transform (other) data.
Examples include samplers, featurizers, preprocessors, and learners.

Design decisions:
* smlb objects are light-weight. Since fitting is potentially
  data-intense, it can not be done in the initializer. Hence
  fitting is done in a separate method.
* Some transformations return Data, some return something else,
  for example, learners can return predictive distributions.
  Therefore, two types of DataTransformation are distinguished:
  transformations from and to Data, and transformations from
  data to something else. The latter would terminate chains of
  transformations: 
  data_1 -A_1-> data_2 -A_2-> ... -> data_n-1 -B-> data_n,
  where A are data-to-data and B are data-to-other transformations.
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Sequence

import numpy as np

from smlb import BenchmarkError, InvalidParameterError
from smlb import Data, complement
from smlb import SmlbObject
from smlb import is_sequence
from smlb import params


class DataTransformation(SmlbObject, metaclass=ABCMeta):
    """Abstract base class for data transformations.

    A DataTransformation is a function object that can be fitted on
    training data and applied to (other) data to return transformed data.
    Its interface contains two functions, fit() and apply().

    Provides:
        fit: accepts data, setting internal state
        apply: accepts and returns transformed data.
    """

    def fit(self, data: Data) -> "DataTransformation":
        """Adjusts internal state based on data ('fitting', 'training').

        Parameters:
            data: training data

        Returns:
            self, to allow chaining

        Example:
            DataTransformation(...).fit(data).apply(otherdata)
        """

        return self

    @abstractmethod
    def apply(self, data: Data) -> Any:
        """Transforms data.

        Parameters:
            data: data to transform

        Returns:
            transformation result; can be derived from Data, but does not have to be
        """

        raise NotImplementedError


class DataValuedTransformation(DataTransformation):
    """Abstract base class for all transformation that return Data.

    The apply() method returns an object derived from Data.
    """

    @abstractmethod
    def apply(self, data: Data) -> Data:
        """Transforms data.

        Parameters:
            data: data to transform

        Returns:
            transformed data
        """

        raise NotImplementedError


class DataPipelineTransformation(DataValuedTransformation):
    """Pipeline of data-valued transformations that are applied sequentially."""

    def __init__(self, steps: Sequence[DataValuedTransformation], *args, **kwargs):
        """Initialize state.

        Parameters:
            steps: List of data-valued transformations
        """
        super().__init__(*args, **kwargs)
        steps = params.sequence(steps, type_=DataValuedTransformation)
        self._steps = steps

    def fit(self, data: Data) -> "DataPipelineTransformation":
        """Fit each step.

        Steps are fit sequentially and used to transform the data.
        The transformed output of each step is used as the training data for the next step.

        Parameters:
            data: training data

        Returns:
            self, to allow chaining
        """
        for step in self._steps:
            data = step.fit(data).apply(data)
        return self

    def apply(self, data: Data) -> Data:
        """Apply each step sequentially.

        The transformed output of each step is used as the training data for the next step.

        Parameters:
            data: data to transform

        Returns:
            transformed data
        """
        for step in self._steps:
            data = step.apply(data)
        return data


class IdentityTransformation(DataValuedTransformation):
    """Returns data unchanged."""

    def apply(self, data: Data) -> Data:
        """Return data unchanged.

        The identity transformation.

        Parameters:
            data: any data

        Returns:
            unchanged data
        """

        return data


class InvertibleTransformation(SmlbObject, metaclass=ABCMeta):
    """Abstract mix-in base class for invertible transformations.

    For DataTransformations that are invertible in the loose sense that
    data -> transformation -> inverse transformation yields data related
    in some way to the original inputs. In particular, inversion does not
    need to be exact, for example, dimensionality reduction might return
    original data points only up to projection onto the learned subspace.

    After fitting, use 'inverse()' to retrieve the inverse data transformation:
    f = transformation(...).fit(training_data);
    f.inverse().apply(f.apply(other_data)) # "close" to identity in some sense

    The inverse transformation must be a DataValuedTransformation.
    """

    @abstractmethod
    def inverse(self) -> DataValuedTransformation:
        """Return inverse of DataTransformation."""

        raise NotImplementedError


class DataTransformationFailureMode:
    """Provide failure mode handling for 1:1 data transformations.

    Provides utility functionality for one-to-one data transformations (mapping one input sample
    to one output sample) to handle failed transformations of individual samples.
    """

    def __init__(self, failmode, num_samples: int):
        """Initialize failure handler.

        Parameters:
            failmode: how to handle failed descriptor calculations, either due to rejected SMILES
                encodings or failing descriptor code. Possible values:
                "raise" [default]: raise a BenchmarkException
                "drop": drop the sample. Returned Data will have fewer samples
                ("mask", mask): where `mask` is a NumPy array with dtype bool whose entries will
                    be set to False for failures
                ("index", index): where `index` is an empty list to which the indices of failed
                    entries will be appended
            num_samples: number of samples that are transformed
        """

        self.num_samples = params.integer(num_samples, from_=0)
        self.failmode = self.failmode(failmode)

        if is_sequence(self.failmode) and self.failmode[0] == "mask":
            self.failmode = "mask"
            if len(failmode[1]) != self.num_samples:
                raise InvalidParameterError(
                    "failure mode mask length of {self.num_samples}", len(self.mask)
                )
            self.mask = failmode[1]
            self.mask.fill(False)

        if is_sequence(self.failmode) and self.failmode[0] == "index":
            self.failmode = "index"
            self.index = failmode[1]

        self.failures = []  # list of indices of failed samples

    @staticmethod
    def failmode(failmode):
        """Failure mode.

        Validate that argument is failure mode, similar to smlb.params.
        See __init__ for valid values.
        """

        ipe = InvalidParameterError("valid failure mode specification", failmode)

        if failmode in ("raise", "drop"):
            return failmode

        if not (is_sequence(failmode) and len(failmode) == 2):
            raise ipe

        if (
            failmode[0] == "mask"
            and isinstance(failmode[1], np.ndarray)
            and failmode[1].ndim == 1
            and failmode[1].dtype.name == "bool"
        ):
            return failmode

        if failmode[0] == "index" and isinstance(failmode[1], list) and len(failmode[1]) == 0:
            return failmode

        raise ipe

    def handle_failure(self, i):
        """Take action according to failure mode.

        Parameters:
            i: index of failed sample
        """

        if self.failmode == "raise":
            raise BenchmarkError(f"DataTransformation failed for sample #{i}")
        elif self.failmode in ("drop", "mask", "index"):
            self.failures.append(i)
        else:
            raise BenchmarkError(f"Internal error, unknown failure mode {self._failmode_failmode}")

    def finalize(self, data: Data) -> Data:
        """Change dataset according to registered failures and failure mode.

        Parameters:
            data: transformed Data

        Returns:
            Transformed Data after handling failures.
        """

        self.failures = sorted(list(set(self.failures)))  # remove duplicate indices

        if self.failmode == "raise":
            if len(self.failures) > 0:
                raise BenchmarkError("DataTransformation failed for some samples")
            return data
        elif self.failmode == "drop":
            return complement(data, data.subset(self.failures))  # todo: duplicates?
        elif self.failmode == "mask":
            self.mask[self.failures] = True
            return data
        elif self.failmode == "index":
            self.index.extend(self.failures)
            return data

        raise BenchmarkError(f"Internal error, unrecognized failure mode '{self.failmode}'")
