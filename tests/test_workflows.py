"""Workflow tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2020, Citrine Informatics.
"""

import pytest

pytest.importorskip("sklearn")

import smlb


#############################
#  LearningCurveRegression  #
#############################


def test_learning_curve_regression():
    """Simple examples"""

    from smlb.datasets.synthetic.friedman_1979.friedman_1979 import Friedman1979Data

    dataset = Friedman1979Data(dimensions=5)

    validation_set = smlb.GridSampler(size=2 ** 5, domain=[0, 1], rng=0)
    training_sizes = [10, 12, 16]
    training_sets = tuple(
        smlb.RandomVectorSampler(size=size, rng=0) for size in training_sizes
    )  # dataset domain is used by default

    from smlb.learners.scikit_learn.gaussian_process_regression_sklearn import (
        GaussianProcessRegressionSklearn,
    )

    learner_gpr_skl = GaussianProcessRegressionSklearn(
        random_state=0
    )  # default is Gaussian kernel
    from smlb.learners.scikit_learn.random_forest_regression_sklearn import (
        RandomForestRegressionSklearn,
    )

    learner_rf_skl = RandomForestRegressionSklearn(random_state=0)

    from workflows.learning_curve_regression import LearningCurveRegression

    workflow = LearningCurveRegression(
        data=dataset,
        training=training_sets,
        validation=validation_set,
        learners=[learner_rf_skl, learner_gpr_skl],
    )  # default evaluation
    workflow.run()
