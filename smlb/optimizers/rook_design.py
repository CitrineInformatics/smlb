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
    """Rook design is a bounded, derivative-free optimizer.
    At each iteration it investigates a large number of options along a large number of orthogonal
    dimensions, then selects the best positions as the seeds for the next iteration ("pruning").

    This implementation is a simple way to test the efficacy of pruning and its dependence on
    both the number of options considered and the maximum step size.
    """

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
        """Initialize state.

        Parameters:
            rng: pseudo-random number generator seed
            max_iters: the maximum number of iterations
            num_seeds: the number of starting points, and the number of points chosen at the end
                of each iteration
            resolution: the number of points to sample along a single dimension for a single seed
            max_relative_jump: the maximum relative step size along a single dimension. If a given
                dimension has length `L` and a seed has value `x` along that dimension, then points
                are drawn by uniformly sampling `resolution` points from the range
                [x - max_relative_jump * L, x + max_relative_jump * L] (clipped by the bounds).
                The value must be on (0, 1]. For a value of 1, the entire range is always sampled.
            dimensions_varied: how many randomly selected dimensions to explore with each step.
                'all' indicates all dimensions. An integer directly specifies the number of
                dimensions. A float on (0, 1) indicates the fractional number of the total.

        TODO: add tolerance stopping conditions
        """
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
        rng = self.random.split(1)[0]
        sampler = RandomVectorSampler(size=self._num_seeds, domain=domain, rng=rng)
        current_seeds = sampler.apply(data)  # get starting seeds

        for _ in range(self._max_iters):
            trial_points = self._make_moves(current_seeds, domain, num_dimensions)
            scores = function_tracker.apply(trial_points)
            current_seeds = self.select_best(trial_points, scores)

    def _determine_num_dimensions(self, total_dimensions: int) -> int:
        """Apply the self._dimensions_varied argument to a total number of dimensions to
        determine the number of dimensions varied with each step.
        """
        if self._dimensions_varied == "all":
            return total_dimensions
        elif isinstance(self._dimensions_varied, float):
            return int(np.ceil(self._dimensions_varied * total_dimensions))
        elif isinstance(self._dimensions_varied, int):
            if self._dimensions_varied > total_dimensions:
                warnings.warn(
                    f"Rook design optimizer attempts to vary {self._dimensions_varied} dimensions "
                    f"with each iteration, but provided dataset only has {total_dimensions} dimensions."
                )
            return min(total_dimensions, self._dimensions_varied)
        else:
            raise BenchmarkError(f"{self._dimensions_varied} is not of acceptable type.")

    def _make_moves(
        self, seeds_table: TabularData, domain: Sequence[Tuple[float, float]], num_dimensions: int
    ) -> TabularData:
        """Produce a set of possible moves from given seed points.

        Parameters:
            seeds_table: the seed points as a tabular data source
            domain: the bounds of the domain, given as a sequence of bounds tuples (lower, upper)
            num_dimensions: the number of dimensions to explore along

        Returns:
            A tabular data source containing all of the possible next-step points
        """
        seeds: np.ndarray = seeds_table.samples()
        total_dimensions = seeds.shape[1]
        # Randomly select dimensions to vary
        dimension_indices = self.random.permutation(range(total_dimensions))[:num_dimensions]
        # For each seed and each dimension generate uniformly-spaced samples, then stack everything
        candidates_array = np.vstack(
            [
                self._move_along_dimension(seed, domain, d_idx)
                for seed in seeds
                for d_idx in dimension_indices
            ]
        )
        # remove duplicates
        return TabularData(candidates_array).subset(duplicates=False)

    def _move_along_dimension(
        self, seed: np.ndarray, domain: Sequence[Tuple[float, float]], d_index: int
    ) -> np.ndarray:
        """Generate an array of points which vary along a single dimension.

        Parameters:
            seed: the starting point, a 1-d array
            domain: the bounds of the domain, given as a sequence of bounds tuples (lower, upper)
            d_index: the index of the dimension to vary
        """
        seed = seed.reshape((1, -1))
        # Make `resolution` copies of the seed point
        candidates = np.tile(seed, (self._resolution, 1))

        # Determine the lower and upper bounds of the range to sample
        value = seed[0, d_index]
        lb, ub = domain[d_index]
        max_jump = (ub - lb) * self._max_relative_jump
        range_lower = max(lb, value - max_jump)
        range_upper = min(ub, value + max_jump)

        # Replace the values along the `d_index` dimension with the trial values
        candidates[:, d_index] = np.linspace(range_lower, range_upper, self._resolution)
        return candidates

    def select_best(self, data: TabularData, scores: Sequence[float]) -> TabularData:
        """Select the best points given a tabular data set and a list of matching scores."""
        best_indices = np.argsort(scores)[:self._num_seeds]
        return data.subset(best_indices)
