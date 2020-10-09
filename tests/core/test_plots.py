"""Plotting tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2019-2020, Matthias Rupp, Citrine Informatics.
"""

import pytest

mpl = pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt

import numpy as np

import smlb

#############################
#  GeneralizedFunctionPlot  #
#############################


def test_GeneralizedFunctionPlot_1():
    """Simple aspects"""

    # accidental passing of results to initializer fails
    with pytest.raises(smlb.InvalidParameterError):
        smlb.GeneralizedFunctionPlot(results=None)


def test_GeneralizedFunctionPlot_2():
    """Test cases"""

    np.random.seed(0)
    a = np.random.normal(2, 0.1, 30)
    b = np.random.uniform(1, 2, 30)
    c = np.random.normal(1.5, 0.2, 30)
    data = [[(1, a), (2, b)], [(1, a), (2, b)]]

    # render points and box-whisker plots with, without and with manual rectification
    gfp = smlb.GeneralizedFunctionPlot(visualization_type=("box-whisker", "points"), rectify=True)
    gfp.evaluate(data)
    gfp.render()

    gfp = smlb.GeneralizedFunctionPlot(visualization_type=("box-whisker", "points"), rectify=False)
    gfp.evaluate(data)
    gfp.render()

    gfp = smlb.GeneralizedFunctionPlot(visualization_type=("box-whisker", "points"), rectify=0.1)
    gfp.evaluate(data)
    gfp.render()

    # three groups
    data = data + [[(1, c)]]
    gfp = smlb.GeneralizedFunctionPlot(visualization_type="box-whisker", rectify=True)
    gfp.evaluate(data)
    gfp.render()


def test_GeneralizedFunctionPlot_3():
    """Special and border cases."""

    # too many groups to rectify must fail only if actually rectifying
    data = [[(1, (1, 2, 3))]] * 99  # too many curves to rectify

    smlb.GeneralizedFunctionPlot(rectify=False).evaluate(data)
    smlb.GeneralizedFunctionPlot(rectify=0.0).evaluate(data)
    with pytest.raises(smlb.InvalidParameterError):
        smlb.GeneralizedFunctionPlot(rectify=True).evaluate(data)
    with pytest.raises(smlb.InvalidParameterError):
        smlb.GeneralizedFunctionPlot(rectify=0.01).evaluate(data)


# todo: test export to files

#######################
#  LearningCurvePlot  #
#######################


def test_LearningCurvePlot_1():
    """Test cases"""

    np.random.seed(0)
    a = np.random.normal(2, 0.1, 30)
    b = np.random.uniform(1, 2, 30)
    c = np.random.normal(1.5, 0.2, 30)
    data = [[(1, a), (2, b)], [(1, a), (2, b)]]

    lcp = smlb.LearningCurvePlot()
    lcp.evaluate(data)
    lcp.render()

    data = data + [[(1, c)]]
    lcp = smlb.LearningCurvePlot(visualization_type="box-whisker", rectify=True)
    lcp.evaluate(data)
    lcp.render()


def test_LearningCurvePlot_Fits():
    """Asymptotic fits."""

    data = [
        [(10 ** 0, 10 ** np.asfarray([5, 4, 6])), (10 ** 10, 10 ** np.asfarray([0, -1, 1, -2, 2]))]
    ]
    lcp = smlb.LearningCurvePlot(fit_lambda=0)
    lcp.evaluate(data)
    fit_data = lcp.auxiliary["asymptotic_fits"][0]
    assert np.allclose([fit_data["offset"], fit_data["slope"]], [5, -0.5], atol=1e-6)

