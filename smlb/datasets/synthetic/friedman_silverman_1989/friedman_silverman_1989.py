"""Synthetic 10-dimensional test function by Friedman & Silverman (1989).

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp 2020, Citrine Informatics.

See class FriedmanSilverman1989Data for details.
"""

import numpy as np

from smlb import params, VectorSpaceData


class FriedmanSilverman1989Data(VectorSpaceData):
    r"""Synthetic 10-dimensional test function by Friedman & Silverman (1989).

    \[ f(x_1,...,x_{10}) = 0.1 \exp( 4 x_1 ) + \frac{4}{ 1 + \exp( - (x_2 - 0.5) / 0.05 ) } + 3 x_3 + 2 x_4 + x5 \]

    where $x \in [0,1]^{10}$.

    This function appears in Section 5.5 on page 17 of

    Jerome H. Friedman, Bernard W. Silverman: Flexible Parsimonious Smoothing and Additive Modeling,
    Technometrics 31(1): 3-21, 1989. DOI 10.2307/1270359

    This implementation introduces some additional flexibility by allowing to specify input dimensionality
    (at least 5), with all variables after the fifth one being ignored.

    For additive independent standard-normal noise as in the publication above,
    use the smlb LabelNoise DataTransformation with NormalNoise:
    data = FriedmanSilverman1989Data(...); noisy_data = LabelNoise(noise=NormalNoise(...)).fit(data).apply(data)
    """

    def __init__(self, dimensions=10, **kwargs):
        """Initialize state.

        Parameters:
            dimensions: dimensionality; at least 5; 10 in original publication; higher dimensions do not change function values

        Raises:
            InvalidParameterError: on invalid parameter values
        """

        dimensions = params.integer(dimensions, from_=5)
        domain = params.hypercube_domain((0, 1), dimensions=dimensions)

        super().__init__(
            dimensions=dimensions,
            function=self.__class__.friedman_silverman_1989,
            domain=domain,
            **kwargs
        )

    @staticmethod
    def friedman_silverman_1989(xx):
        """Computes Friedman & Silverman (1989) test function without noise.

        Parameters:
            xx: matrix, rows are input vectors

        Returns:
            vector of computed function values
        """

        xx = params.real_matrix(xx)  # base class verifies dimensionality and domain

        return (
            0.1 * np.exp(4 * xx[:, 0])
            + 4 / (1 + np.exp(-(xx[:, 1] - 0.5) / 0.05))
            + 3 * xx[:, 2]
            + 2 * xx[:, 3]
            + xx[:, 4]
        )

    # properties 'dimensions' and 'domain' are provided by base classes
