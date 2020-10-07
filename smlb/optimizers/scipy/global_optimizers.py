"""Global optimizers as implemented in Scipy.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
2019-2020, Citrine Informatics.
"""

from typing import Optional, Sequence, Tuple
from abc import abstractmethod

from scipy.optimize import dual_annealing, OptimizeResult

from smlb import (
    params,
    Random,
    TabularData,
    VectorSpaceData,
    Optimizer,
    TrackedTransformation,
)


class ScipyGlobalOptimizer(Optimizer, Random):
    """An abstract class for Scipy global optimizers."""

    @abstractmethod
    def _minimization_algorithm(
        self, func: callable, bounds: Sequence[Tuple[float, float]], seed: int
    ) -> OptimizeResult:
        """The algorithm that performs the minimization.

        Parameters:
            func: The objective function to be minimized. Must be in the form f(x), where x is
                the argument in the form of a 1-D array.
            bounds: bounds on the elements of x, specified as a list of tuples (min, max) for each
                dimension. It is required that len(bounds) == len(x).
            seed: seed for the pseudo-random number generator.

        Returns:
            An instance of OptimizeResult with information about the optimization trajectory.
        """
        raise NotImplementedError

    def _minimize(self, data: VectorSpaceData, function_tracker: TrackedTransformation):
        func = lambda x: function_tracker.apply(TabularData(x.reshape(1, -1)))
        bounds = data.domain
        seed = self.random.split(1)  # split off a new random seed each time `optimize` is called
        # TODO: include a callback to record the results of each iteration. Store this info in
        #   TrackedTransformation and include it when creating the OptimizationTrajectory.
        self._minimization_algorithm(func, bounds, seed)


class ScipyDualAnnealingOptimizer(ScipyGlobalOptimizer):
    """Scipy's Dual Annealing optimizer.

    Dual annealing alternates between simulated annealing to explore the global space
    and a local optimization method to efficiently find local minima.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html#scipy.optimize.dual_annealing
    """

    def __init__(
        self,
        rng: int = None,
        maxiter: int = 1000,
        local_search_options: Optional[dict] = None,
        initial_temp: float = 5230.0,
        restart_temp_ratio: float = 2e-05,
        visit: float = 2.62,
        accept: float = -5.0,
        maxfun: int = 1e7,
        no_local_search: bool = False,
        **kwargs
    ):
        """Initialize state.

        Scipy-specific parameters are passed through.

        Parameters:
            rng: integer seed. Will be used to generate a new seed each time the optimizer is run.
            maxiter: The maximum number of iterations, where one iteration is one round of
                simulated annealing followed by one use of a local optimizer to find a local min.
            local_search_options: an optional kwargs dictionary to pass to the local minimizer,
                scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
                If no args are passed then the minimizer defaults to the L-BFGS-B method, since
                the problems being studied have bounds but no constraints.
            initial_temp: The initial temperature, use higher values to facilitates a wider search
                and more easily escape local minima.
            restart_temp_ratio: The temperature, relative to the initial temperature, at which
                the annealing process restarts.
            visit: a parameter of the visiting distribution. A higher value corresponds to a
                heavier tail and longer potential jumps.
            accept: a parameter of the acceptance distribution. A lower value means that uphill
                moves are less likely to be accepted.
            maxfun: soft limit for the total number of function evaluation calls.
            no_local_search: if true then the local search step is skipped, and this reduces
                 to a generalized simulated annealing optimizer.
        """
        super().__init__(rng=rng, **kwargs)

        self._maxiter = params.integer(maxiter, from_=1)
        self._local_search_options = local_search_options or {}  # TODO: verify dictionaries
        self._initial_temp = params.real(initial_temp, above=0.01, to=5e4)
        self._restart_temp_ratio = params.real(restart_temp_ratio, above=0.0, below=1.0)
        self._visit = params.real(visit, above=0.0, to=3.0)
        self._accept = params.real(accept, above=-1e4, to=-5.0)
        self._maxfun = params.integer(maxfun, from_=1)
        self._no_local_search = params.boolean(no_local_search)

    def _minimization_algorithm(
        self, func: callable, bounds: Sequence[Tuple[float, float]], seed: int
    ) -> OptimizeResult:
        return dual_annealing(
            func,
            bounds,
            seed=seed,
            maxiter=self._maxiter,
            local_search_options=self._local_search_options,
            initial_temp=self._initial_temp,
            restart_temp_ratio=self._restart_temp_ratio,
            visit=self._visit,
            accept=self._accept,
            maxfun=self._maxfun,
            no_local_search=self._no_local_search,
        )
