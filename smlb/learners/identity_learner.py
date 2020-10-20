"""An identity learner that reproduces a synthetic function.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
(c) 2019-20, Citrine Informatics.
"""

import numpy as np

from smlb import (
    params,
    Learner,
    Data,
    VectorSpaceData,
    NormalPredictiveDistribution,
    InvalidParameterError,
)


class IdentityLearner(Learner):
    """A trivial learner that reproduces a provided function."""

    def __init__(self, function: VectorSpaceData, **kwargs):
        super().__init__(**kwargs)
        self._function = params.instance(function, VectorSpaceData)

    def apply(self, data: Data):
        if not data.is_finite:
            raise InvalidParameterError(
                "a finite dataset", f"an infinite dataset of type {data.__class__}"
            )
        means = self._function.labels(data.samples())
        stddevs = np.zeros_like(means)
        return NormalPredictiveDistribution(means, stddevs)
