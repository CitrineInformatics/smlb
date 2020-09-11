"""Metrics (performance functions) tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.
"""

import pytest

import numpy as np
import numpy.testing
import scipy as sp

import smlb

###############
#  Interface  #
###############


def test_interface_erroneous_arguments():
    """Tests whether errors are raised for wrong keyword arguments."""

    # EvaluationMetric can not be instantiated because it is abstract

    # this test also ensures that EvaluationMetric.__init__ calls super().__init__
    with pytest.raises(Exception):
        smlb.AbsoluteResiduals(orientt=+1)  # spelling error


##########################
#  Multiple-Class Tests  #
##########################


def test_orient():
    """Test orient argument for oriented metrics."""

    classes = (
        smlb.MeanAbsoluteError,
        smlb.MeanSquaredError,
        smlb.RootMeanSquaredError,
        smlb.MeanLogPredictiveDensity,
        smlb.MeanContinuousRankedProbabilityScore,
        smlb.UncertaintyCorrelation,
    )

    true = smlb.NormalPredictiveDistribution([1, 2, 3], [0.5, 0.6, 0.7])
    pred = smlb.NormalPredictiveDistribution([1.1, 2.2, 2.9], [0.4, 0.7, 0.65])
    for c in classes:
        resa, resb = c(orient=-1)(true, pred), c(orient=+1)(true, pred)
        if smlb.is_sequence(resa):
            assert (resa == -resb).all(), c.__name__
        else:
            assert resa == -resb, c.__name__

        with pytest.raises(Exception):
            c(orientt=-1)  # ensure misspelt argument raises


###############
#  Residuals  #
###############


def test_residuals_examples():
    """Test residuals for simple example."""

    residuals = smlb.Residuals()

    assert (residuals([-1, 0, 1], [-2, 1, 1]) == [-1, 1, 0]).all()


#######################
#  AbsoluteResiduals  #
#######################


def test_absolute_residuals_examples():
    """Test absolute residuals for simple example."""

    abs_residuals = smlb.AbsoluteResiduals()

    assert (abs_residuals([-1, 0, 1], [-2, 1, 1]) == [+1, 1, 0]).all()


######################
#  SquaredResiduals  #
######################


def test_squared_residuals_examples():
    """Test squared residuals for simple example."""

    squared_residuals = smlb.SquaredResiduals()

    assert (squared_residuals([-1, 0, 1], [-2, -2, 1]) == [1, 4, 0]).all()


#######################
#  MeanAbsoluteError  #
#######################


def test_mae_examples():
    """Test MAE for simple example."""

    mae = smlb.MeanAbsoluteError()

    assert mae([-1, 2], [0, 9]) == 4


#######################
#   MeanSquaredError  #
#######################


def test_mse_examples():
    """Test MSE for simple example."""

    rmse = smlb.MeanSquaredError()

    assert rmse([-1, 2], [0, 9]) == 25


###########################
#   RootMeanSquaredError  #
###########################


def test_rmse_examples():
    """Test RMSE for simple example."""

    rmse = smlb.RootMeanSquaredError()

    assert rmse([-1, 2], [0, 9]) == 5


######################################
#  StandardizedRootMeanSquaredError  #
######################################


def test_std_rmse_examples():
    """Test stdRMSE with simple examples."""

    # no bias correction by default
    srmse = smlb.StandardizedRootMeanSquaredError()
    assert srmse([-1, 2], [0, 9]) == 10 / 3

    # not enough samples to compute variance
    with pytest.raises(smlb.InvalidParameterError):
        srmse([], [])

    # not enough samples to compute variance
    with pytest.raises(smlb.InvalidParameterError):
        srmse([1,], [2,])

    # not enough variance in labels
    with pytest.raises(smlb.InvalidParameterError):
        srmse([-1, -1.001], [10, 11])  # threshold: 0.001

    # bias correction
    srmse = smlb.StandardizedRootMeanSquaredError(bias_correction=1.5)
    assert srmse([-1, 2], [0, 9]) == 5 / 3


##########################
#  LogPredictiveDensity  #
##########################


def test_lpd_examples():
    """Test logarithmized predictive density for simple example."""

    lpd = smlb.LogPredictiveDensity()
    true = [0, 1, 2, 2]
    pred_mu = [0, 2, 4, 4]
    pred_sigma = [1, 1, 1, 2]
    res = np.log(
        [
            sp.stats.norm.pdf(x, loc=mu, scale=sigma)
            for x, mu, sigma in zip(true, pred_mu, pred_sigma)
        ]
    )

    np.testing.assert_allclose(lpd(true, (pred_mu, pred_sigma)), res, atol=1e-7)


##############################
#  MeanLogPredictiveDensity  #
##############################


def test_mlpd_examples():
    """Test mean logarithmized predictive density for simple example."""

    mlpd = smlb.MeanLogPredictiveDensity()
    true = [0, 1, 2, 2]
    pred_mu = [0, 2, 4, 4]
    pred_sigma = [1, 1, 1, 2]
    res = np.mean(
        np.log(
            [
                sp.stats.norm.pdf(x, loc=mu, scale=sigma)
                for x, mu, sigma in zip(true, pred_mu, pred_sigma)
            ]
        )
    )

    np.testing.assert_allclose(mlpd(true, (pred_mu, pred_sigma)), res, atol=1e-7)


######################################
#  ContinuousRankedProbabilityScore  #
######################################


def test_crps_examples():
    """Test CRPS against reference calculation in Mathematica.

    Absolute tolerance is set to the difference between numerical integration and closed-form expression.
    """

    crps = smlb.ContinuousRankedProbabilityScore()

    np.testing.assert_allclose(
        crps((1, 2, 3), ((1.5, 1.5, 3), (0.5, 0.5, 0.1))),
        (0.3012206788138085, 0.30122056014259146, 0.023369497725510956),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        crps((-2.5, -1.25, 0.1), ((3.7, 0, -0.1), (10, 0.1, 1))),
        (3.8231838217254834, 1.193580730690417, 0.24959975842919802),
        atol=1e-5,
    )


##########################################
#  MeanContinuousRankedProbabilityScore  #
##########################################


def test_mcrps_examples():
    """Test mCRPS against CRPS."""

    crps = smlb.ContinuousRankedProbabilityScore()
    mcrps = smlb.MeanContinuousRankedProbabilityScore()

    np.testing.assert_allclose(
        mcrps((1, 2, 3), ((1.5, 1.5, 3), (0.5, 0.5, 0.1))),
        np.mean(crps((1, 2, 3), ((1.5, 1.5, 3), (0.5, 0.5, 0.1)))),
        atol=1e-6,
    )


#########################
#  Standard Confidence  #
#########################


def test_sc_examples():
    """ Test standard confidence. """
    
    true = [0, 1, 2, 2]
    pred_mu = [0, 2, 4, 4]
    pred_sigma = [1, 1.25, 1.75, 2.2]
    sc = smlb.metrics.StandardConfidence()
    np.testing.assert_allclose(sc(true, (pred_mu, pred_sigma)), 0.75)


#########################################
#  RootMeanSquareStandardizedResiduals  #
#########################################


def test_rmsse_examples():
    """ Test root mean square standardized residuals. """
    
    true = [0, 1, 2, 2]
    pred_mu = [0, 2, 4, 4]
    pred_sigma = [1, 1, 1, 2]
    rmsse = smlb.RootMeanSquareStandardizedResiduals()
    np.testing.assert_allclose(rmsse(true, (pred_mu, pred_sigma)), 1.225, rtol=0.01)


############################
#  UncertaintyCorrelation  #
############################


def test_uccorr_examples():
    """ Test root mean square standardized residuals. """
    
    true = [0, 1, 2, 2]
    pred_mu = [0, 2, 4, 4]
    pred_sigma = [1, 1, 1, 2]
    uccorr = smlb.UncertaintyCorrelation()
    np.testing.assert_allclose(uccorr(true, (pred_mu, pred_sigma)), 0.522, rtol=0.01)


######################################################
#  TwoSampleCumulativeDistributionFunctionStatistic  #
######################################################


def test_cdfs_example_normal():
    """Test two-sample cumulative distribution function statistic for normal distribution.
    
    This is a statistical test.
    """

    n = 10000
    np.random.seed(0)
    sample_a = np.random.normal(size=n)
    sample_b = np.random.normal(size=n)

    np.testing.assert_allclose(
        smlb.two_sample_cumulative_distribution_function_statistic(sample_a, sample_b),
        0,
        atol=1e-3,
    )

    np.testing.assert_allclose(
        smlb.two_sample_cumulative_distribution_function_statistic(
            sample_a, sample_b, f=lambda p, t: np.abs(p - t)
        ),
        0,
        atol=1e-1,
    )


def test_cdfs_example():
    """Test two-sample cumulative distribution function statistic for simple example."""

    # example with same y-scales

    sample_a = [1, 1, 3]
    sample_b = [2, 3, 3]

    np.testing.assert_allclose(
        smlb.two_sample_cumulative_distribution_function_statistic(
            sample_a, sample_b, f=lambda p, t: np.abs(p - t)
        ),
        2 / 3 + 1 / 3,
        atol=1e-6,
    )

    # example with differing y-scales

    sample_a = [1, 2, 3, 4]
    sample_b = [2, 4, 4]

    np.testing.assert_allclose(
        smlb.two_sample_cumulative_distribution_function_statistic(
            sample_a, sample_b, f=lambda p, t: np.square(p - t)
        ),
        (1 / 4) ** 2 + (2 / 4 - 1 / 3) ** 2 + (3 / 4 - 1 / 3) ** 2,
        atol=1e-6,
    )

    # same example with different integralf

    np.testing.assert_allclose(
        smlb.two_sample_cumulative_distribution_function_statistic(
            sample_a, sample_b, f=lambda p, t: np.abs(p - t), g=lambda s, w: np.max(s)
        ),
        3 / 4 - 1 / 3,
        atol=1e-6,
    )


def test_cdfs_sign_example():
    """Test signedness of two-sample cumulative distribution function statistic."""

    sample_a = [1, 2]
    sample_b = [2]

    assert smlb.two_sample_cumulative_distribution_function_statistic(
        sample_a, sample_b, f=lambda p, t: p - t, g=lambda s, w: np.sign(s) * np.abs(s)
    )

    assert smlb.two_sample_cumulative_distribution_function_statistic(
        sample_a, sample_b, f=lambda p, t: p - t, g=lambda s, w: np.sign(s) * np.abs(s)
    ) == -smlb.two_sample_cumulative_distribution_function_statistic(
        sample_b, sample_a, f=lambda p, t: p - t, g=lambda s, w: np.sign(s) * np.abs(s)
    )
