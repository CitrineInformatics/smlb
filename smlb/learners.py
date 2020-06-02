"""Machine learning algorithms.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
Citrine Informatics 2019.
"""

from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np

from smlb.parameters import params
from smlb.data import Data
from smlb.transformations import DataTransformation
from smlb.distributions import PredictiveDistribution

# todo: add hyperparameters to interface


class Learner(DataTransformation, metaclass=ABCMeta):
    """Abstract base class for machine-learning algorithms.
    
    `Learner`s return `PredictiveDistribution`s.
    
    Deterministic learners can return a `DeltaPredictiveDistribution`.
    For example, (deterministic) principal component analysis could return a
    multi-variate delta predictive distribution.

    A `Learner` can provide all, some, or none of the possible decompositions
    of its `PredictiveDistribution`.
    """

    def fit(self, data: Data) -> "Learner":
        """Fits the model to training data.

        Parameters:
            data: training data
        
        Returns:
            self (allows chaining)
        """

        return self

    @abstractmethod
    def apply(self, data: Data) -> PredictiveDistribution:
        """Applies fitted model to predict new inputs.

        Parameters:
            data: prediction data

        Returns:
            a predictive distribution
        """

        raise NotImplementedError


class UnsupervisedLearner(Learner):
    """Abstract base class for unsupervised machine-learning algorithms."""

    pass


class SupervisedLearner(Learner):
    """Abstract base class for supervised machine learning algorithms."""

    def fit(self, data: Data) -> Learner:
        """Fits the model to training data.

        Parameters:
            data: labeled training data

        Returns:
            self (allows chaining)
        
        Raises:
            InvalidParameterError if data is not labeled
        """

        data = params.instance(data, Labels)
        if not data.is_labeled:
            raise InvalidParameterError("Labeled data", "unlabeled data")

        return self
