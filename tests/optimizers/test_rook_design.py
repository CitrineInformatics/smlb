"""Tests of rook design optimizer

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
(c) 2019-20, Citrine Informatics.
"""

from smlb import ExpectedValue, TrackedTransformation

from smlb.datasets.synthetic.schwefel26_1981.schwefel26_1981 import Schwefel261981Data
from smlb.optimizers.rook_design import RookDesignOptimizer
from smlb.learners.identity_learner import IdentityLearner


def test_foo():
    dims = 4
    data = Schwefel261981Data(dimensions=dims)
    learner = IdentityLearner(data)
    scorer = ExpectedValue()
    func = TrackedTransformation(learner, scorer, maximize=False)

    optimizer = RookDesignOptimizer(rng=0, max_iters=5, num_seeds=2, resolution=16, dimensions_varied=2)

    trajectory = optimizer.optimize(data, func)
    print("foo")