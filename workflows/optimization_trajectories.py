"""Optimization trajectories for multiple optimization algorithms on a single response surface.
"""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from smlb import (
    Workflow,
    VectorSpaceData,
    params,
    Learner,
    Scorer,
    Optimizer,
    TrackedTransformation,
    OptimizerIteration
)


class OptimizationTrajectory(Workflow):
    """
    """

    def __init__(
            self,
            training_data: VectorSpaceData,
            learner: Learner,
            scorer: Scorer,
            optimizers: Sequence[Optimizer]
    ):
        self._training_data = params.instance(training_data, VectorSpaceData)
        self._scorer = params.instance(scorer, Scorer)
        learner = params.instance(learner, Learner)
        self._func = TrackedTransformation(learner, self._scorer)
        self._optimizers = params.sequence(optimizers, type_=Optimizer)

    def run(self):
        """Execute workflow."""
        self._learner.fit(self._training_data)

        for optimizer in self._optimizers:
            results: Sequence[OptimizerIteration] = \
                optimizer.optimize(self._training_data, self._func)

            best_score_so_far = results[0].score
            best_score = np.empty(len(results))
            for i, optimization_iter in enumerate(results):
                if optimization_iter.score > best_score_so_far:
                    best_score_so_far = optimization_iter.score
                best_score[i] = best_score_so_far * self._func.direction

            plt.plot(range(len(best_score)), best_score)

        plt.show()
