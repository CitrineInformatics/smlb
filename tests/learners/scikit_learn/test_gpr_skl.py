"""GaussianProcessRegressionSklearn tests.

Scientific Machine Learning Benchmark:
A benchmark of regression models in chem- and materials informatics.
"""

import pytest

import numpy as np

skl = pytest.importorskip("sklearn")

import smlb
from smlb.learners.scikit_learn.gaussian_process_regression_sklearn import GaussianProcessRegressionSklearn


def test_GaussianProcessRegressionSklearn_1():
    """Simple examples."""

    # linear function with linear kernel
    kernel = skl.gaussian_process.kernels.DotProduct(sigma_0=0, sigma_0_bounds="fixed")
    gpr = GaussianProcessRegressionSklearn(kernel=kernel, optimizer=None, rng=1)
    train_data = smlb.TabularData(data=np.array([[-1], [1]]), labels=np.array([-1, 1]))
    valid_data = smlb.TabularData(data=np.array([[-2], [-1], [0], [1], [2]]))
    preds = gpr.fit(train_data).apply(valid_data)
    mean, stddev = preds.mean, preds.stddev

    assert np.allclose(mean, [-2, -1, 0, 1, 2])
    assert stddev[0] > stddev[1] > stddev[2] < stddev[3] < stddev[4]


def test_GaussianProcessRegressionSklearn_2():
    """All predictive distributions.

    Linear noise-free function, linear kernel + white noise kernel.
    The optimized noise level is expected to go to its lower bound.
    """

    kernel = skl.gaussian_process.kernels.DotProduct(
        sigma_0=0, sigma_0_bounds="fixed"
    ) + skl.gaussian_process.kernels.WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1))
    gpr = GaussianProcessRegressionSklearn(kernel=kernel, rng=1)
    n = 100
    data = np.reshape(np.linspace(start=1, stop=n, num=n) * 2, (n, 1))
    labels = np.linspace(start=1, stop=n, num=n) * 3
    train_data = smlb.TabularData(data=data, labels=labels)
    valid_data = smlb.TabularData(data=train_data.samples())
    preds = gpr.fit(train_data).apply(valid_data)

    assert preds.has_signal_part and preds.has_noise_part
    conf, noise = preds.signal_part, preds.noise_part

    assert np.allclose(conf.mean, train_data.labels())
    assert np.allclose(conf.stddev, np.ones(n) * np.sqrt(1e-5), atol=1e-3)

    assert (preds.mean == conf.mean).all()
    assert np.allclose(preds.stddev, np.ones(n) * np.sqrt(np.square(conf.stddev) + 1e-5))

    assert np.allclose(noise.mean, np.zeros(shape=n))
    assert np.allclose(noise.stddev, np.sqrt(1e-5))


def test_GaussianProcessRegressionSklearn_3():
    """All predictive distributions.

    Linear noisy function, linear kernel + white noise kernel.
    The optimized noise level is expected to go to its true value.
    """

    kernel = skl.gaussian_process.kernels.DotProduct(
        sigma_0=0, sigma_0_bounds="fixed"
    ) + skl.gaussian_process.kernels.WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e5))
    gpr = GaussianProcessRegressionSklearn(kernel=kernel, rng=1)
    n, nlsd = 100, 0.5
    data = smlb.TabularData(data=np.ones(shape=(n, 1)) * 2, labels=np.ones(shape=n) * 3)
    data = smlb.LabelNoise(noise=smlb.NormalNoise(stddev=nlsd, rng=1)).fit(data).apply(data)
    preds = gpr.fit(data).apply(data)

    assert preds.has_signal_part and preds.has_noise_part
    conf, noise = preds.signal_part, preds.noise_part

    assert np.allclose(conf.mean, np.ones(n) * 3, atol=1e-1)
    assert np.allclose(conf.stddev, np.ones(n) * nlsd, atol=1e-1)

    assert (preds.mean == conf.mean).all()
    assert np.allclose(preds.stddev, np.sqrt(np.square(conf.stddev) + np.square(nlsd)), atol=1e-1)

    assert np.allclose(noise.mean, np.zeros(shape=n))
    assert np.allclose(noise.stddev, nlsd, atol=1e-1)
