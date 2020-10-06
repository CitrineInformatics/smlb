"""Model for the potential energy of two atoms by Philip M. Morse (1929).

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.

See class MorsePotentialData for details.
"""

import numpy as np

from smlb import params, VectorSpaceData


class MorsePotentialData(VectorSpaceData):
    r"""Model for the potential energy of two atoms by Philip M. Morse (1929).

    Based on:
    Philip M. Morse: Diatomic Molecules According to the Wave Mechanics. II. Vibrational Levels,
    Physical Review 34(1): 57-64, 1929. DOI 10.1103/PhysRev.34.57

    This dataset implements equation 4:

    \[ V(r) = D \exp( -2a(r-r_0) ) - 2D \exp( -a(r-r_0) ) \]

    where $-D$ is the value of the single minimum at $r=r_0$ and $V(\infty) = 0$.
    This is equivalent to $V(r) = D \gamma (\gamma - 2)$ for $\gamma = \exp( -a(r-r_0) )$.
    """

    def __init__(self, D, r0, a, domain=(0, np.inf), **args):
        """Initialize state.
        
        Parameters:
            D: potential parameter determining well depth -D
            r0: potential parameter determining location r0 of minimum
            a: potential parameter, where 1/a is proportional to well width
            domain: domain of dataset; defaults to unit [0,inf) on which the potential is defined
            All parameters from base class 'ComputedLabelsVectorSpaceData' initializer

        Raises:
            InvalidParameterError: on invalid parameter values
        """

        self._d = params.real(D, above=0)
        self._r0 = params.real(r0, above=0)
        self._a = params.real(a, above=0)

        def morsef(r):
            """Evaluate Morse potential at a sequence of vectors r.
            
            Parameters:
                r: n x 1 matrix of n one-dimensional vectors

            Returns:
                vector of Morse potential values at r
            """

            r = params.real_matrix(r, ncols=1)
            n = len(r)

            gamma = np.exp(-self._a * (r - self._r0))
            v = self._d * (np.square(gamma) - 2 * gamma)
            return v.reshape(n)

        super().__init__(dimensions=1, function=morsef, domain=domain, **args)

    @property
    def D(self):
        """Negative well depth parameter of the potential.

        Returns:
            Parameter D
        """

        return self._d

    @property
    def r0(self):
        """Location of minimum parameter of the potential.

        Returns:
            Parameter r0
        """

        return self._r0

    @property
    def a(self):
        """Inverse well width scale parameter of potential.

        Returns:
            Parameter a.
        """

        return self._a
