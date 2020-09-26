"""Optimization trajectories for multiple optimization algorithms on a single response surface.
"""

from typing import Sequence, Optional

import numpy as np

from smlb import (
    Data,
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
    """Optimization trajectories for multiple trials and multiple optimizers on a single model.

    Parameters:
        data: the real-valued vector space that defines the problem
        model: any function that can be evaluated on the vector space, whether a regression
            model or an analytic function
        scorer: score the predictions supplied by the model
        optimizers: sequence of optimizers, each of which tries to find the point in `data`
            that optimizes the score produced by `scorer`
        num_trials: number of trials to perform for each optimizer
        training_data: optional data on which to train the model (unnecessary if the model
            is pre-trained or is an analytic function)
    """

    def __init__(
            self,
            data: VectorSpaceData,
            model: Learner,
            scorer: Scorer,
            optimizers: Sequence[Optimizer],
            num_trials: int = 1,
            training_data: Optional[Data] = None
    ):
        self._data = params.instance(data, VectorSpaceData)
        self._scorer = params.instance(scorer, Scorer)
        self._model = params.instance(model, Learner)
        self._optimizers = params.sequence(optimizers, type_=Optimizer)
        self._num_trials = params.integer(num_trials, from_=1)
        self._training_data = params.optional_(
            training_data, lambda arg: params.instance(arg, Data)
        )

    def run(self):
        """Execute workflow."""
        if self._training_data is not None:
            self._model.fit(self._training_data)
        func = TrackedTransformation(self._model, self._scorer)

        num_optimizers = len(self._optimizers)
        trajectories = np.empty(
            (num_optimizers, self._num_trials), dtype=Sequence[OptimizerIteration]
        )
        best_score_trajectory = np.empty_like(trajectories, dtype=Sequence[float])

        for i, optimizer in enumerate(self._optimizers):
            for j in range(self._num_trials):
                results = optimizer.optimize(self._data, func)
                trajectories[i, j] = results
                # TODO: Break this out into a new type of metric object
                #   Are there any other quantities we might want to calculate?
                best_score_trajectory[i, j] = self.best_score_trajectory(results, func.direction == 1.0)

        # TODO: add Evaluations to plot the results

    @staticmethod
    def best_score_trajectory(results: Sequence[OptimizerIteration], maximize: bool) -> Sequence[float]:
        num_scores = np.sum([len(result.scores) for result in results])
        best_score = np.empty(num_scores)
        idx = 0
        best_score_so_far = results[0].scores[0]
        direction = 1.0 if maximize else -1.0

        for optimization_iter in results:
            for eval in optimization_iter.scores:
                if eval > best_score_so_far:
                    best_score_so_far = eval
                best_score[idx] = best_score_so_far * direction
                idx += 1
        return best_score
