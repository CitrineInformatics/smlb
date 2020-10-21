"""Tests of the learning curve regression workflow.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2020, Citrine Informatics.
"""

import pytest

pytest.importorskip("sklearn")

import smlb


def test_learning_curve_regression():
    """Simple examples"""

    from smlb.datasets.synthetic import Friedman1979Data

    dataset = Friedman1979Data(dimensions=5)

    validation_set = smlb.GridSampler(size=2 ** 5, domain=[0, 1], rng=0)
    training_sizes = [10, 12, 16]
    training_sets = tuple(
        smlb.RandomVectorSampler(size=size, rng=0) for size in training_sizes
    )  # dataset domain is used by default

    from smlb.learners import GaussianProcessRegressionSklearn

    learner_gpr_skl = GaussianProcessRegressionSklearn(rng=0)  # default is Gaussian kernel
    from smlb.learners import RandomForestRegressionSklearn

    learner_rf_skl = RandomForestRegressionSklearn(rng=0)

    from smlb.workflows import LearningCurveRegression

    workflow = LearningCurveRegression(
        data=dataset,
        training=training_sets,
        validation=validation_set,
        learners=[learner_rf_skl, learner_gpr_skl],
    )  # default evaluation
    workflow.run()
