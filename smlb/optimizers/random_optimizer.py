"""An "optimizer" that draws random samples.

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
2019-2020, Citrine Informatics.
"""

from typing import Optional, Any

from smlb import (
    params,
    Optimizer,
    Random,
    RandomVectorSampler,
    VectorSpaceData,
    TrackedTransformation,
)


class RandomOptimizer(Optimizer, Random):
    """Draws random samples.

    Parameters:
        num_samples: the number of random samples to draw
        domain: optional domain from which to draw values. If not provided, then the
            optimization domain is taken to be that of `data` parameter passed to `optimize()`.
        rng: pseudo-random number generator
    """

    def __init__(self, num_samples: int, domain: Optional[Any] = None, rng=None, **kwargs):
        super().__init__(rng=rng, **kwargs)
        self._num_samples = params.integer(num_samples, above=0)
        self._sampler = RandomVectorSampler(size=self._num_samples, domain=domain, rng=rng)

    def _minimize(self, data: VectorSpaceData, function_tracker: TrackedTransformation):
        """Generate num_samples random samples and evaluate them."""
        samples = self._sampler.apply(data)
        function_tracker.apply(samples)
