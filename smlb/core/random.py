"""Pseudo-random numbers.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.

Supports reproducability when using pseudo-random numbers 
in non-deterministic environments, for example, parallel 
and distributed computing. See class `Random` for details.

Correct and comprehensive usage of this module should enable reproducible
benchmarks in the sense that re-running a benchmark should deterministically
yield the same results, given the same version of the underlying pseudo-
random number generator (here, NumPy.random). Differing runs can be achieved
by varying the initial seed for the pseudo-random number generator.
"""

import numpy as np

from smlb import InvalidParameterError
from smlb import SmlbObject
from smlb import params

# the current implementation is a NumPy-based coarse approximation of
# https://github.com/google/jax/blob/master/design_notes/prng.mdbased


class PseudoRandomNumberGenerator:
    """Pseudo-random number generator."""

    def __init__(self, seed):
        """Initializes state of pseudo-random number generator.

        Parameters:
            seed: key to initialize pseudo-random number generator
        """

        seed = params.integer(seed, from_=0, to=2 ** 32 - 1)
        self._random = np.random.RandomState(seed=seed)

    def split(self, num=2):
        """Splits pseudo-random number generator key into several independent keys.

        Parameters:
            num: number of resultant keys

        Returns:
            num pseudo-random number generator keys
        """

        return self._random.randint(low=1, high=2 ** 32, size=num)

    def __getattr__(self, name):
        """Forward method calls to underlying pseudo-random number generator.

        Parameters:
            name: method name

        Returns:
            corresponding NumPy pseudo-random number generator instance method
        """

        return getattr(self._random, name)


class Random(SmlbObject):
    """Mix-in base class providing pseudo-random numbers.

    Derive from this class if using pseudo-random numbers.

    Design:
        https://github.com/google/jax/blob/master/design_notes/prng.md

    Counter-based pseudo-random number generation:
        John K. Salmon, Mark A. Moraes, Ron O. Dror, David E. Shaw:
        Parallel Random Numbers: As Easy As 1, 2, 3. In: Proceedings
        of the International Conference for High Performance Computing,
        Networking, Storage and Analysis (SC~11), Seattle, Washington,
        November 12--18, 2011. DOI 10.1145/2063384.2063405

    Splittable pseudo-random number generators:
        Koen Claessen, Micha{\l} H. Pa{\l}ka: Splittable Pseudorandom
        Number Generators using Cryptographic Hashing, p. 47--58. In:
        The 18th {ACM} {SIGPLAN} International Conference on Functional
        Programming (ICFP 2013), Boston, Massachusetts, September 25--27,
        2013. DOI 10.1145/2578854.2503784

    Interface:
        random - pseudo-random number generator
    """

    def __init__(self, rng=None, **kwargs):
        """Initialize state.

        Parameters:
            rng: seed (key) for pseudo-random number generator.
                 This parameter must be specified to encourage correc usage
                 of pseudo-random numbers throughout the benchmark.
        """

        super().__init__(**kwargs)

        if rng is None:
            raise InvalidParameterError(
                "rng seed", "nothing", "pseudo-random number generator seed must be specified"
            )
        rng = params.integer(rng, from_=0, to=2 ** 32 - 1)
        self._random = PseudoRandomNumberGenerator(seed=rng)

    @property
    def random(self):
        """Access to pseudo-random generator functionality.

        Returns:
            object providing pseudo-random number generator functionality.
            In particular, split() and drawing pseudo-random numbers.
        """

        return self._random
