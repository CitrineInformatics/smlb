"""Synthetic d-dimensional test function by Schwefel (1981).

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
2020, Citrine Informatics.

See class Schwefel1981Data for details.
"""

import numpy as np

from smlb import params, VectorSpaceData


class Schwefel261981Data(VectorSpaceData):
    r"""Synthetic d-dimensional test function by Schwefel (1981).

    \[ f(x_1,...,x_d) = 418.9829 d - \sum_{i=1}^d x_i \sin( \sqrt(|x_i|) ) \]

    where $d$ is input space dimensionality and $x \in [-500, 500]^d$.

    This function is attributed to

    Hans-Paul Schwefel: Numerical Optimization of Computer Models, Wiley, Chichester, 1981.
    Volume 26 of Interdisciplinary Systems Research series.

    Multiple optimization test functions by H.-P. Schwefel exist and are in use.
    For disambiguation, this function has been named Schwefel 26 (1981).

    Unfortunately, the original publication was not available to us, preventing verification.
    """

    def __init__(self, dimensions: int, **kwargs):
        """Initialize Schwefel 26 test function.

        Parameters:
            dimensions: dimensionality of the problem

        Raises:
            InvalidParameterError: on invalid parameter values

        Examples:
            __init__(dimension=2)
        """

        dimensions = params.integer(dimensions, above=0)
        domain = params.hypercube_domain((-500, 500), dimensions=dimensions)

        super().__init__(
            dimensions=dimensions, function=self.__class__.schwefel26_1981, domain=domain, **kwargs
        )

    @staticmethod
    def schwefel26_1981(xx):
        """Computes Schwefel (1981) test function 26.

        Parameters:
            xx: input matrix, rows are samples

        Returns:
            sequence of computed labels

        Examples:
            schwefel26_1981(np.random.uniform(-500, 500, (100,2))) # evaluate on 100 2-dimensional inputs
        """

        xx = params.real_matrix(xx)  # base class verifies dimensionality and domain
        d = xx.shape[1]

        return 418.9829 * d - np.sum(np.multiply(xx, np.sin(np.sqrt(np.abs(xx)))), axis=-1)

    # properties 'dimensions' and 'domain' are provided by base classes
