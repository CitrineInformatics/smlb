"""Synthetic 6-dimensional test function by Friedman (1979).

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp 2020, Citrine Informatics.

See class Friedman1979Data for details.
"""

import numpy as np

from smlb import params, VectorSpaceData


class Friedman1979Data(VectorSpaceData):
    r"""Synthetic 6-dimensional test function by Friedman (1979).

    \[ f(x_1,...,x_6) = 10 \sin(\pi x_1 x_2) + 20 (x_3 - 1/2)^2 + 10 x_4 + 5 x_5 + 0 x_6 \]

    where $x \in [0,1]^6$.

    This function originally appeared as Equation (22) on page 17 of

    Jerome H. Friedman: A Tree-Structured Approach to Nonparametric Multiple Regression, pp. 5-22 in
    Thomas Gasser, Murray Rosenblatt: Smoothing Techniques for Curve Estimation, Springer, 1979.
    Volume 757 in Lecture Notes in Mathematics. DOI 10.1007/BFb0098486

    and was later used again on page 294 of

    Jerome H. Friedman, Eric Grosse, Werner Stuetzle: Multidimensional Additive Spline Approximation,
    SIAM Journal on Scientific and Statistical Computing 4(2): 291-301, SIAM, 1983. DOI 10.1137/0904023

    This implementation introduces some additional flexibility by allowing to specify input dimensionality
    (at least 5), with all variables after the fifth one being ignored.

    The original formulation included additive independent standard-normal noise. For this, use the
    smlb LabelNoise DataTransformation with NormalNoise:
    data = Friedman1979Data(...); noisy_data = LabelNoise(noise=NormalNoise(...)).fit(data).apply(data)
    """

    def __init__(self, dimensions=6, **kwargs):
        """Initialize state.

        Parameters:
            dimensions: dimensionality; at least 5; 6 in original paper; higher dimensions do not change function

        Raises:
            InvalidParameterError: on invalid parameter values
        """

        dimensions = params.integer(dimensions, from_=5)
        domain = params.hypercube_domain((0, 1), dimensions=dimensions)

        super().__init__(
            dimensions=dimensions, function=self.__class__.friedman1979, domain=domain, **kwargs
        )

    @staticmethod
    def friedman1979(xx):
        """Computes Friedman (1979) test function without noise term

        Parameters:
            xx: sequence of vectors

        Returns:
            sequence of computed labels
        """

        xx = params.real_matrix(xx)  # base class verifies dimensionality and domain

        return (
            10 * np.sin(np.pi * xx[:, 0] * xx[:, 1])
            + 20 * np.power(xx[:, 2] - 1 / 2, 2)
            + 10 * xx[:, 3]
            + 5 * xx[:, 4]
        )

    # properties 'dimensions' and 'domain' are provided by base classes
