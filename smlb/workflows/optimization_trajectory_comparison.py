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
    OptimizerTrajectory,
    Evaluation,
    OptimizationTrajectoryPlot,
)


class OptimizationTrajectoryComparison(Workflow):
    """Optimization trajectories for multiple trials and multiple optimizers on a single model.
    The same class of optimizer could be represented multiple times with different parameters.

    Parameters:
        data: the real-valued vector space that defines the problem
        model: any function that can be evaluated on the vector space, whether a regression
            model or an analytic function
        scorer: score the predictions supplied by the model
        optimizers: sequence of optimizers, each of which tries to find the point in `data`
            that optimizes the score produced by `scorer`
        evaluations: one or more evaluations to apply;
            default is the median trajectory for each optimizer
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
        evaluations: Sequence[Evaluation] = (OptimizationTrajectoryPlot(),),
        num_trials: int = 1,
        training_data: Optional[Data] = None,
    ):
        self._data = params.instance(data, VectorSpaceData)
        self._scorer = params.instance(scorer, Scorer)
        self._model = params.instance(model, Learner)
        self._optimizers = params.sequence(optimizers, type_=Optimizer)
        self._evaluations = params.tuple_(
            evaluations, lambda arg: params.instance(arg, Evaluation)
        )
        self._num_trials = params.integer(num_trials, from_=1)
        self._training_data = params.optional_(
            training_data, lambda arg: params.instance(arg, Data)
        )

    def run(self):
        """Execute workflow.

        1. Run each optimizer once for each trial, creating a matrix of `OptimizerTrajectory` objects.
        2. For each optimizer calculate the "best score trajectory" for each trial and coerce
            them into the format required by Evaluation objects.
            TODO: there's a potential abstraction here, similar to the Metric in
                LearningCurveRegression. For example, we could examine individual trajectories.
        3. Apply evaluations to the results.
        """
        if self._training_data is not None:
            self._model.fit(self._training_data)
        func = TrackedTransformation(self._model, self._scorer)

        num_optimizers = len(self._optimizers)
        trajectories = np.empty((num_optimizers, self._num_trials), dtype=OptimizerTrajectory)

        for i, optimizer in enumerate(self._optimizers):
            for j in range(self._num_trials):
                results: OptimizerTrajectory = optimizer.optimize(self._data, func)
                trajectories[i, j] = results

        def _collect_optimization_results(list_of_trajectories: Sequence[OptimizerTrajectory]):
            """Convert optimization results into the format required by Evaluation objects.

            Parameters:
                list_of_trajectories: a sequence of `OptimizerTrajectory` objects, each one
                    a separate trial of the same optimizer.

            Returns:
                A sequence of tuples of the form (int, Sequence[float]), where the first entry
                if the evaluation number (horizontal axis) and the second entry is a sequence
                of all the scores at that point (vertical axis).
            """
            max_trajectory_length = np.max([t.num_evaluations for t in list_of_trajectories])
            best_score_trajectories = np.vstack(
                [
                    t.best_score_trajectory(func.maximize, max_trajectory_length)
                    for t in list_of_trajectories
                ]
            )

            return [(j + 1, best_score_trajectories[:, j]) for j in range(max_trajectory_length)]

        eval_data = [
            _collect_optimization_results(trajectories[i, :]) for i in range(len(self._optimizers))
        ]

        for eval_ in self._evaluations:
            eval_.evaluate(eval_data)
            eval_.render()
