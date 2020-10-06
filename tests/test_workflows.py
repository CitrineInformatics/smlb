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

    from smlb.workflows.learning_curve_regression import LearningCurveRegression

    workflow = LearningCurveRegression(
        data=dataset,
        training=training_sets,
        validation=validation_set,
        learners=[learner_rf_skl, learner_gpr_skl],
    )  # default evaluation
    workflow.run()


def test_optimization_trajectories():
    """Ensure that a simple optimization workflow can be run."""
    from smlb.datasets.synthetic.friedman_1979.friedman_1979 import Friedman1979Data
    dataset = Friedman1979Data(dimensions=5)
    sampler = smlb.RandomVectorSampler(size=100, rng=0)
    training_data = sampler.fit(dataset).apply(dataset)

    from smlb.learners.scikit_learn.random_forest_regression_sklearn import RandomForestRegressionSklearn

    learner = RandomForestRegressionSklearn(uncertainties="naive", random_state=0)
    learner.fit(training_data)

    pi_scorer = smlb.ProbabilityOfImprovement(target=2, goal="minimize")

    from smlb.core.optimizer import RandomOptimizer
    optimizer = RandomOptimizer(num_samples=30, rng=0)

    from smlb.workflows.optimization_trajectories import OptimizationTrajectory
    workflow = OptimizationTrajectory(
        data=dataset,
        model=learner,
        scorer=pi_scorer,
        optimizers=[optimizer, optimizer],  # just to check that it can handle multiple optimizers
        num_trials=3
    )
    workflow.run()
