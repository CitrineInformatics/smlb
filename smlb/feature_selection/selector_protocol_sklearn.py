from abc import abstractmethod
from typing import Protocol

import numpy as np
from numpy import typing as npt


class SelectorProtocolSklearn(Protocol):
    """Protocol that defines the methods expected from a scikit-learn feature selector."""

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "SelectorProtocolSklearn":
        """Fit the underlying estimator to the given data.

        Parameters:
            X: training data of shape ``(n_samples, n_features)``
            y: target values of shape ``(n_samples,)`` or ``(n_samples, n_targets)``
        """
        raise NotImplementedError

    @abstractmethod
    def get_support(self, indices: bool = False) -> np.array:
        """
        Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : If True, the return value will be an array of integers, rather than a boolean mask.
            Default is False.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If ``indices`` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention.
            If ``indices`` is True, this is an integer array of shape
            [# output features] whose values are indices into the input feature vector.
        """
        raise NotImplementedError
