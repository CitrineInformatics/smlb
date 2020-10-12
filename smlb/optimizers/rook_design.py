"""Optimize using rook design.

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
2019-2020, Citrine Informatics.
"""

from typing import Union, Sequence, Tuple
import warnings

import numpy as np

from smlb import (
    params,
    Random,
    RandomVectorSampler,
    TabularData,
    VectorSpaceData,
    Optimizer,
    TrackedTransformation,
    BenchmarkError,
)


class RookDesignOptimizer(Optimizer, Random):
    """Rook design is a bounded, derivative-free optimizer."""

    def __init__(
        self,
        rng: int = None,
        max_iters: int = 50,
        num_seeds: int = 1,
        resolution: int = 64,
        max_relative_jump: float = 1.0,
        dimensions_varied: Union[str, float, int] = "all",
        **kwargs,
    ):
        super().__init__(rng=rng, **kwargs)

        self._max_iters = params.integer(max_iters, from_=1)
        self._num_seeds = params.integer(num_seeds, from_=1)
        self._resolution = params.integer(resolution, from_=2)
        self._max_relative_jump = params.real(max_relative_jump, above=0.0, to=1.0)
        self._dimensions_varied = params.any_(
            dimensions_varied,
            lambda arg: params.integer(arg, above=0),
            lambda arg: params.real(arg, above=0.0, below=1.0),
            lambda arg: params.enumeration(arg, {"all"}),
        )

    def _minimize(self, data: VectorSpaceData, function_tracker: TrackedTransformation):
        num_dimensions = self._determine_num_dimensions(data.dimensions)
        domain = data.domain
        seed = self.random.split(1)[0]
        sampler = RandomVectorSampler(size=self._num_seeds, domain=domain, rng=seed)
        current_seeds = sampler.apply(data)

        for _ in range(self._max_iters):
            trial_points = self._make_moves(current_seeds, domain, num_dimensions)
            scores = function_tracker.apply(trial_points)
            current_seeds = self.select_best(trial_points, scores)

    def _determine_num_dimensions(self, data_dimensions) -> int:
        if self._dimensions_varied == "all":
            return data_dimensions
        elif isinstance(self._dimensions_varied, float):
            return np.ceil(self._dimensions_varied * data_dimensions)
        elif isinstance(self._dimensions_varied, int):
            if self._dimensions_varied > data_dimensions:
                warnings.warn(
                    f"Rook design optimizer attempts to vary {self._dimensions_varied} dimensions "
                    f"with each iteration, but provided dataset only has {data_dimensions} dimensions."
                )
            return min(data_dimensions, self._dimensions_varied)
        else:
            raise BenchmarkError(f"{self._dimensions_varied} is not of acceptable type.")

    def _make_moves(self, seeds_table: TabularData, domain: Sequence[Tuple[float, float]], num_dimensions: int) -> TabularData:
        seeds: np.ndarray = seeds_table.samples()
        total_dimensions = seeds.shape[1]
        dimension_indices = self.random.shuffle(range(total_dimensions))[:num_dimensions]
        candidates_array = np.vstack([self._move_along_dimension(seed, domain, d) for seed in seeds for d in dimension_indices])
        return TabularData(candidates_array)

    def _move_along_dimension(self, seed: np.ndarray, domain: Sequence[Tuple[float, float]], d_index: int) -> np.ndarray:
        seed = seed.reshape((1, -1))
        value = seed[d_index]
        lb = domain[d_index][0]
        ub = domain[d_index][1]
        step_size = (ub - lb) * self._max_relative_jump
        range_lower = max(lb, value - step_size)
        range_upper = min(ub, value + step_size)
        trial_points = np.linspace(range_lower, range_upper, self._resolution)
        candidates = np.tile(seed, (self._resolution, 1))
        candidates[:, d_index] = trial_points

    def select_best(self, data: TabularData, scores: Sequence[float]) -> TabularData:
        best_indices = np.argsort(scores)[:self._num_seeds]
        return data.samples(best_indices)
