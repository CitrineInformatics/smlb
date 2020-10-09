"""Tests of the optimization trajectory comparison workflow.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
2020, Citrine Informatics.
"""

import pytest

pytest.importorskip("sklearn")

import smlb


def test_optimization_trajectories():
    """Ensure that a simple optimization workflow can be run."""
    from smlb.datasets.synthetic.friedman_1979.friedman_1979 import Friedman1979Data
    dataset = Friedman1979Data(dimensions=5)
    sampler = smlb.RandomVectorSampler(size=100, rng=0)
    training_data = sampler.fit(dataset).apply(dataset)

    from smlb.learners.scikit_learn.random_forest_regression_sklearn import RandomForestRegressionSklearn

    learner = RandomForestRegressionSklearn(uncertainties="naive", rng=0)
    learner.fit(training_data)

    pi_scorer = smlb.ProbabilityOfImprovement(target=2, goal="minimize")

    from smlb.core.optimizer import RandomOptimizer
    optimizer = RandomOptimizer(num_samples=30, rng=0)

    from smlb.workflows.optimization_trajectory_comparison import OptimizationTrajectoryComparison
    workflow = OptimizationTrajectoryComparison(
        data=dataset,
        model=learner,
        scorer=pi_scorer,
        optimizers=[optimizer, optimizer],  # just to check that it can handle multiple optimizers
        num_trials=3
    )
    workflow.run()
