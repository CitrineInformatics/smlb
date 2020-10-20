"""Tests of rook design optimizer

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
(c) 2019-20, Citrine Informatics.
"""
import pytest

import numpy as np

from smlb import ExpectedValue, TrackedTransformation, InvalidParameterError

from smlb.datasets.synthetic.schwefel26_1981.schwefel26_1981 import Schwefel261981Data
from smlb.optimizers.rook_design import RookDesignOptimizer
from smlb.learners.identity_learner import IdentityLearner


@pytest.fixture(scope='module')
def data():
    num_dimensions = 4
    return Schwefel261981Data(dimensions=num_dimensions)


@pytest.fixture(scope='module')
def func(data) -> TrackedTransformation:
    learner = IdentityLearner(data)
    scorer = ExpectedValue()
    return TrackedTransformation(learner, scorer, maximize=False)


def test_single_iteration(data, func):
    """Check that the expected number of candidates are explored in a single iteration."""
    num_iters = 1
    num_seeds = 2
    resolution = 16
    fractional_dims = 0.5
    dimensions_varied = int(data.dimensions * fractional_dims)
    optimizer = RookDesignOptimizer(
        rng=0,
        max_iters=num_iters,
        num_seeds=num_seeds,
        resolution=resolution,
        dimensions_varied=fractional_dims
    )
    trajectory = optimizer.optimize(data, func)
    assert trajectory.num_evaluations == num_seeds * dimensions_varied * resolution * num_iters


def test_multiple_iterations(data, func):
    """Test that the best scores from each iteration become seeds for the next iteration.
    Note that this is only strictly true when the jump size can span the entire domain.
    If the jumps are small then there is no fixed grid and the candidates in a given step
    might not _exactly_ include a seed.
    """
    num_iters = 5
    num_seeds = 2
    resolution = 8
    optimizer = RookDesignOptimizer(
        rng=0,
        max_iters=num_iters,
        num_seeds=num_seeds,
        resolution=resolution
    )
    trajectory = optimizer.optimize(data, func)

    for i in range(num_iters - 1):
        old_scores = trajectory.steps[i].scores
        best_scores = np.sort(old_scores)[:num_seeds]
        new_scores = trajectory.steps[i+1].scores
        for good_score in best_scores:
            assert good_score in new_scores

    expected_candidates_per_step = num_seeds * resolution * data.dimensions
    # Subsequent iterations are likely to have duplicates so the actual number of evaluations
    # is lower than expected_candidates_per_step * num_iters
    assert trajectory.num_evaluations < expected_candidates_per_step * num_iters


def test_short_moves(data, func):
    """Test that the new seeds aren't very far away when using a small max_relative_jump."""
    num_iters = 1
    num_seeds = 1
    resolution = 8
    max_relative_jump = 0.01
    lb, ub = data.domain[0]
    max_jump = max_relative_jump * (ub - lb)  # domain is uniform for Schwefel function

    optimizer = RookDesignOptimizer(
        rng=0,
        max_iters=num_iters,
        num_seeds=num_seeds,
        resolution=resolution,
        max_relative_jump=max_relative_jump
    )
    trajectory = optimizer.optimize(data, func)
    candidates = trajectory.steps[0]._input.samples()

    # Calculate the Manhattan distances between the (arbitrary) first candidate and all other
    # candidates. The max should be no more than max_jump * 2 (if the two candidates are on
    # opposite sides of the seed).
    distances = np.sum(np.abs(candidates - candidates[0]), axis=1)
    assert np.max(distances) <= max_jump * 2


def test_rook_design_parameters():
    """Test more complicated parameter validation."""
    with pytest.raises(InvalidParameterError):
        RookDesignOptimizer()  # no rng seed

    with pytest.raises(InvalidParameterError):
        RookDesignOptimizer(rng=0, max_relative_jump=0.0)  # jump size cannot be 0

    with pytest.raises(InvalidParameterError):
        RookDesignOptimizer(rng=0, dimensions_varied=0)  # dimensions varied cannot be 0

    with pytest.raises(InvalidParameterError):
        RookDesignOptimizer(rng=0, dimensions_varied=-3)  # dims varied cannot be negative

    with pytest.raises(InvalidParameterError):
        RookDesignOptimizer(rng=0, dimensions_varied="foo")  # not a valid keyword
