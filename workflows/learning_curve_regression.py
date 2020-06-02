"""Learning curve for multiple regression learners on a single dataset.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
2020, Matthias Rupp, Citrine Informatics.
"""

from typing import Callable, Optional, Sequence

import numpy as np

from smlb import (
    BenchmarkError,
    Data,
    DataValuedTransformation,
    Evaluation,
    Features,
    IdentityFeatures,
    intersection,
    InvalidParameterError,
    LearningCurvePlot,
    params,
    PredictiveDistribution,
    RootMeanSquaredError,
    ScalarEvaluationMetric,
    Sampler,
    SupervisedLearner,
    Workflow,
)

# todo: regression is currently not query-able for a learner

# todo: add featurization

# todo: no automatic filenames
#       default filenames should be provided that include evaluation type and metric

# todo: support multiple metrics via re-specification of the workflow with identical
#       initialization parameters except the metric. caching of result runs should
#       then enable re-evaluation with different metric without any new calculations.

# todo: the featurizer is currently assumed to not be parameterized, that is, training
#       is a no-op. To allow parameterizable featurizers (for example, from deep learning),
#       it might be necessary to featurize the validation set differently for each training
#       set. Under covariate shift, both training inputs and validation inputs (but not labels)
#       might have to be used for featurizer training.


class LearningCurveRegression(Workflow):
    """Learning-curve for multiple regression learners on a single dataset.

    Algorithm:
    1) Validation data
       Draw validation data from dataset
       For finite datasets, remove validation data from dataset
    2) Training sets
       Draw training sets from remaining dataset
       Validate that there is no overlap with the validation set
    3) Featurization
       Featurize validation and training sets
    4) Training and prediction
       Train each learner on each training set
       For each trained learner, predict validation set
    5) Evaluate results
       Compute evaluation metric for each run
       Render each evaluation

    Current limitations:
    * no hyperparameter optimization
    * no repeated sampling
    """

    def __init__(
        self,
        data: Data,
        training: Sequence[Sampler],
        validation: Sampler,
        learners: Sequence[SupervisedLearner],
        features: DataValuedTransformation = IdentityFeatures(),
        metric: ScalarEvaluationMetric = RootMeanSquaredError(),
        evaluations: Sequence[Evaluation] = (LearningCurvePlot(),),  # todo: add table
        progressf: Optional[Callable[[int, int], None]] = None,
    ):
        """Initialize workflow.

        Parameters:
            data: labeled data
            training: sequence of Samplers, one for each training set size
            validation: Sampler for validation set
            learners: sequence of supervised regression algorithms
            features: any data-valued transformation
            metric: evaluation metric to use; root mean squared error by default
            evaluations: one or more evaluations; default are learning curve and table
            progressf: callable with two parameters, done iterations and total number of iterations
        """

        self._data = params.instance(data, Data)  # todo: params.data(..., is_labeled=True)
        if not self._data.is_labeled:
            raise InvalidParameterError("labeled data", "unlabeled data")
        self._training = params.sequence(training, type_=Sampler)
        self._validation = params.instance(validation, Sampler)
        self._learners = params.sequence(learners, type_=SupervisedLearner)
        self._features = params.instance(features, Features)
        self._metric = params.instance(metric, ScalarEvaluationMetric)
        self._evaluations = params.tuple_(
            evaluations, lambda arg: params.instance(arg, Evaluation)
        )
        self._progressf = params.optional_(
            progressf, lambda arg: params.callable(arg, num_pos_or_kw=2)
        )
        if self._progressf is None:
            self._progressf = lambda *args: None

    def run(self):
        """Execute workflow."""

        nlearn, ntrain = len(self._learners), len(self._training)
        ntotal = nlearn * ntrain
        self._progressf(0, ntotal)

        # 1) Validation data

        # sample validation data from dataset
        validation_data = self._validation.fit(self._data).apply(self._data)

        # remove validation data from dataset for finite datasets
        if self._data.is_finite:
            remaining_data = self._data.complement(validation_data)
        else:  # infinite
            # any finite subset has measure zero
            remaining_data = self._data

        # 2) Training sets

        # sample training sets from remaining dataset
        training_data = tuple(
            sampler.fit(remaining_data).apply(remaining_data) for sampler in self._training
        )

        # verify that the intersection between validation and all training sets is empty
        for train in training_data:
            # this assumes that both validation and training set are finite
            inters = intersection(train, validation_data)
            if inters.num_samples > 0:
                i, j, k = inters.num_samples, validation_data.num_samples, train.num_samples
                msg = f"Non-empty intersection between validation and training data ({i} shared samples out of {j} and {k})"
                raise BenchmarkError(msg)

        # 3) Featurization

        # featurize validation and training sets
        validation_data = self._features.fit(validation_data).apply(validation_data)
        training_data = tuple(self._features.fit(train).apply(train) for train in training_data)

        # 4) Training and prediction

        # train each learner on each training set and predict validation set
        predictions = np.empty((nlearn, ntrain), dtype=PredictiveDistribution)
        for i, learner in enumerate(self._learners):
            for j, training in enumerate(training_data):
                learner.fit(training)
                predictions[i, j] = learner.apply(validation_data)

                self._progressf(i * ntrain + j + 1, ntotal)  # 1-based

        # 5) Evaluate results

        # compute evaluation metric for each run
        metric = np.asfarray(
            [
                [
                    self._metric.evaluate(true=validation_data.labels(), pred=predictions[i, j])
                    for j in range(ntrain)
                ]
                for i in range(nlearn)
            ]
        )

        # render each evaluation
        eval_data = [
            [(train.num_samples, (metric[i, j],)) for j, train in enumerate(training_data)]
            for i, learner in enumerate(self._learners)
        ]
        for eval_ in self._evaluations:
            eval_.evaluate(eval_data)
            eval_.render()
