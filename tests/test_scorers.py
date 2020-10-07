"""Tests of Scorer implementations.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
(c) 2019-20, Citrine Informatics.
"""

import pytest

import numpy as np
from smlb import (
    ProbabilityOfImprovement,
    InvalidParameterError,
    DeltaPredictiveDistribution,
    NormalPredictiveDistribution
)


def test_interface_erroneous_arguments():
    """Test whether errors are raised for invalid keyword arguments."""
    with pytest.raises(InvalidParameterError):
        ProbabilityOfImprovement("not a number")

    with pytest.raises(InvalidParameterError):
        ProbabilityOfImprovement(0.0, "optimize")


def test_likelihood_of_improvement_calculation():
    """Test the accuracy of the 'likelihood of improvement' calculation."""
    configs = [
        (DeltaPredictiveDistribution([1.0, 5.0]),
         ProbabilityOfImprovement(2.0, "maximize"), [0.0, 1.0]),
        (DeltaPredictiveDistribution([1.0, 5.0]),
         ProbabilityOfImprovement(2.0, "minimize"), [1.0, 0.0]),
        (NormalPredictiveDistribution([0.0, 4.0], [2.0, 1.0]),
         ProbabilityOfImprovement(2.0, "maximize"), [0.159, 0.977]),
        (NormalPredictiveDistribution([0.0, 4.0], [2.0, 1.0]),
         ProbabilityOfImprovement(2.0, "minimize"), [0.841, 0.023])
    ]

    for dist, scorer, result in configs:
        np.testing.assert_allclose(scorer.apply(dist), result, atol=1E-3)
