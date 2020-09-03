"""Evaluation metrics.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.

Related terms: objective functions, loss functions, cost functions, 
reward functions, utility functions, fitness functions, score functions, merit functions.

Provides classes EvaluationMetric, ScalarEvaluationMetric, VectorEvaluationMetric.
See documentation for relationships and derived metrics.
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from warnings import warn

import numpy as np
import scipy as sp
import scipy.stats  # for normal distribution. Python 3.8 will offer a 'statistics' module including PDF and CDF of the normal distribution

from .exceptions import InvalidParameterError
from .object import SmlbObject
from .parameters import params

##################
#  Base classes  #
##################


class EvaluationMetric(SmlbObject, metaclass=ABCMeta):
    """Abstract base class for evaluation metrics.
    
    Base class for ScalarEvaluationMetric and VectorEvaluationMetric.

    Design notes:
    * Derived classes define _evaluate(). Actual evaluation is done by evaluate(),
      which can take additional action, for example, modifying the sign of the
      returned value according to a preferred orientation for ScalarEvaluationMetrics.
    * This solution avoids errors due to derived classes' implementations of
      evaluate() not running additional processing required. it does not prevent
      a class from accidentally overriding evaluate() instead of _evaluate().
    * (_)evaluate methods get passed only the observed ('true') labels of the
      validation set. In particular, they do not have access to the training set
      labels. This is because the performance of predictions on a set V should
      not depend on any other external information; including the training set.
      Otherwise, performance on V could change without any change in V.
    """

    # A variant with only evaluate() was tried where each evaluate() returns
    # a call to a processing method, `return self.processingf(result)`.
    # However, for inheritance chains EvaluationMetric -> A -> B
    # this would require an additional parameter 'raw' telling when and when not
    # to modify the result (or more complicated solutions) and was therefore abandoned.

    @abstractmethod
    def _evaluate(self, true, pred):
        """Evaluate metric for given observations and predictions.

        See evaluate() for function signature and explanation.

        Derived classes overwrite this function instead of evaluate()
        to allow further modification by EvaluationMetric class.
        """

        raise NotImplementedError

    def evaluate(self, true, pred):
        """Evaluate metric for given observations and predictions.

        Parameters:
            true: observed property distributions (PredictiveDistribution)
            pred: predictive property distributions (PredictiveDistribution)

        Returns:
            value of evaluation metric; type depends on the evaluation metric,
            for example, a scalar (ScalarEvaluationMetric) or a vector (VectorEvaluationMetric)

        Note that both true and pred are distributions, including but not limited to
        the delta distribution (deterministic values) and the normal distribution.

        Each EvaluationMetric should support at least all combinations (for true and pred) of
        deterministic values (delta distributions) and normal distributions.
        """

        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Provide convenient evaluation by being callable.
        
        See evaluate() for details.
        """

        return self.evaluate(*args, **kwargs)


class ScalarEvaluationMetric(EvaluationMetric):
    """Base class for scalar-valued EvaluationMetrics."""

    def __init__(self, orient=None, **kwargs):
        """Initialize state.

            Parameters:
                orient: actively orients metric towards minimization (-1) or maximization (+1)
                        if unspecified, the natural orientation of the metric is retained

            Raises:
                InvalidParameterError if trying to orient a metric with no natural orientation
            """

        super().__init__(**kwargs)

        orient = params.enumeration(orient, {-1, +1, None})

        self._sign = +1  # default value leaves _evaluate() unchanged
        if orient is not None:
            if not self.has_orientation:
                raise InvalidParameterError("oriented metric", self.orientation)
            # -1 if desired and actual orientation disagree, otherwise +1
            self._sign = orient * self.orientation

    @property
    def has_orientation(self):
        """True if oriented.

        Here, oriented means that the metric has a preferred direction
        (either more negative or more positive values indicating improvement) 
        and is ordered. 
        
        Returns:
            True if the metric has an orientation, False otherwise
        """

        return self.orientation != 0

    @property
    def orientation(self):
        """Whether optimization for this metric means minimization, maximization or neither.

        Examples without orientation include signed residuals and composite metrics.

        Orientation must be constant, that is, it must not change over the lifetime of an object.
        
        Returns:
            -1 for minimization, +1 for maximization, or 0 if not applicable
        """

        return 0  # default is non-oriented, override method to add orientation

    def evaluate(self, true, pred):
        """Evaluate metric for given observations and predictions.

        Parameters:
            true: observed property distributions (PredictiveDistribution)
            pred: predictive property distributions (PredictiveDistribution)

        Returns:
            a scalar value

        Note that both true and pred are distributions, including but not limited to
        the delta distribution (deterministic values) and the normal distribution.

        The desired orientation can be set in the initializer.
        """

        return self._sign * self._evaluate(true, pred)


# todo: introduce a 'summaryf' parameter to enable mean, min, max, ... of vector-valued evaluation metrics


class VectorEvaluationMetric(EvaluationMetric):
    """Base class for vector-valued EvaluationMetrics."""

    def evaluate(self, true, pred):
        """Evaluate metric for given observations and predictions.

        Parameters:
            true: observed property distributions (PredictiveDistribution)
            pred: predicted property distributions (PredictiveDistribution)

        Returns:
            a vector

        Note that both true and pred are distributions, including but not limited to
        the delta distribution (deterministic values) and the normal distribution.
        """

        return self._evaluate(true, pred)


######################
#  Error statistics  #
######################


class Residuals(VectorEvaluationMetric):
    r"""Signed errors (residuals).

    Prediction error residuals $f(x_i) - y_i$, 
    where $x_i$ are inputs, $f$ is the learner and $y_i$ are observed values.
    """

    def _evaluate(self, true, pred):
        """Evaluate prediction error residuals.

        residuals = predicted - observed

        Parameters:
            true: observed property distribution; requires only means
            pred: predictive property distribution; requires only means

        Returns:
            residuals as NumPy array
        """

        true = params.distribution(true).mean
        pred = params.distribution(pred).mean

        return pred - true


class AbsoluteResiduals(Residuals):
    """Absolute value of residuals.
    
    Unsigned residuals. Absolute prediction error residuals $|f(x_i) - y_i|$, 
    where $x_i$ are inputs, $f$ is the learner and $y_i$ are observed values.
    """

    def _evaluate(self, true, pred):
        """Evaluate unsigned prediction errors.

        unsigned residuals = | pred - observed |

        Parameters:
            true: observed property distribution; requires only means
            pred: predictive property distribution; requires only means
        
        Returns:
            unsigned residuals as NumPy array
        """

        return np.abs(super()._evaluate(true, pred))


class SquaredResiduals(Residuals):
    """Squared prediction errors.

    As Residuals, but squared.
    """

    def _evaluate(self, true, pred):
        """Evaluate squared prediction errors.

        squared residuals = ( pred - observed )^2
        
        Parameters:
            true: observed property distribution; requires only means
            pred: predictive property distribution; requires only means
        
        Returns:
            squared residuals as NumPy array
        """

        return np.square(super()._evaluate(true, pred))


class MeanAbsoluteError(ScalarEvaluationMetric):
    """Mean Absolute Error (MAE)."""

    @property
    def orientation(self):
        """Indicate minimization."""

        return -1

    def _evaluate(self, true, pred):
        r"""Evaluate Mean Absolute Error (MAE).

        \[ \text{MAE} = \frac{1}{n} \sum_{i=1}^n | f(x_i) - y_i | \]

        MAE = mean( | pred - observed | )

        Parameters:
            true: observed property distribution; requires only means
            pred: predictive property distribution; requires only means
        
        Returns:
            mean absolute error as floating point number
        """

        return float(np.mean(AbsoluteResiduals()._evaluate(true, pred)))


class MeanSquaredError(ScalarEvaluationMetric):
    """Mean squared error (MSE)."""

    @property
    def orientation(self):
        """Indicate minimization."""

        return -1

    def _evaluate(self, true, pred):
        r"""Mean Squared Error (MSE).

        \[ \text{MSE} = \frac{1}{n} \sum_{i=1}^n ( f(x_i) - y_i )^2 \]

        MSE = mean( square( pred - observed ) )

        Parameters:
            true: observed property distribution; requires only means
            pred: predictive property distribution; requires only means

        Returns:
            mean squared error as a floating point number
        """

        return float(np.mean(SquaredResiduals()._evaluate(true, pred)))


class RootMeanSquaredError(MeanSquaredError):
    """Root Mean Squared Error (RMSE)."""

    # same orientation as MeanSquaredError base class

    def _evaluate(self, true, pred):
        r"""Root Mean Squared Error (RMSE).

        \[ \text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^n ( f(x_i) - y_i )^2 } \]

        MSE = root( mean( square( pred - observed ) ) )

        Parameters:
            true: observed property distribution; requires only means
            pred: predictive property distribution; requires only means

        Returns:
            root mean squared error as a floating point number
        """

        return float(np.sqrt(super()._evaluate(true, pred)))


class StandardizedRootMeanSquaredError(RootMeanSquaredError):
    r"""Standardized Root Mean Squared Error (stdRMSE).
    
    The standardized RMSE (stdRMSE), relative RMSE, or non-dimensional model
    error (NDME) is given by

    stdRMSE = RMSE / std. dev., where

    \[ \text{std. dev.} = \sqrt{ \frac{1}{n} \sum_{i=1}^n ( y_i - \bar{y} )^2 } \]

    and $\bar{y} = \frac{1}{n} \sum_{i=1}^n y_i$. 

    The denominator can be interpreted as the RMSE of a model that predicts the
    mean of the validation set (!) labels. stdRMSE is a unit-less (non-dimensional) 
    quantity, often between 0 (perfect model) and 1 (guess-the-mean performance). 

    If the IID assumption is violated, that is, label distributions of
    training and validation set differ, stdRMSE can be arbitrarily high.

    The name "standardized RMSE" was chosen over "non-dimensional model error"
    because it is more specific (e.g., which "error"?) and more directly related
    to statistical terminology (e.g., "standard score").

    If the IID assumption holds, stdRMSE can be used to compare prediction errors
    across different datasets on the same scale (the datasets can still vary in
    how hard they are to learn).

    An advantage of stdRMSE over RMSE divided by label range is that stdRMSE is less
    statistically volatile (min and max are extremal statistics with high variance).

    For the estimator of the standard deviation, no bias correction is used by default 
    (easing comparisons in many cases). See __init__ docstring for other options.
    """

    def __init__(self, bias_correction: float = 0, **kwargs):
        """Initialize metric.

        Parameters:
            bias_correction: no correction by default. if a positive value d is given,
                division is by n-d. Bessel's correction (d=1) is unbiased for variance
                estimators, but not for standard deviation estimators. While there is
                no value that works across all distributions, d=1.5 is a reasonably
                good correction.
        """

        self._bias_correction = params.real(bias_correction, from_=0)

        super().__init__(**kwargs)

    # same orientation as RootMeanSquaredError

    def _evaluate(self, true, pred):
        """Root mean squared error divided by standard deviation of labels.

        stdRMSE = RMSE / std. dev.

        See class docstring for details.

        Parameters:
            true: observed property distribution; requires only means
            pred: predictive property distribution; requires only means

        Returns:
            standardized root mean squared error as a floating point number
        """

        true = params.distribution(true)

        # ensure sufficiently many samples
        n = len(true.mean)
        if n <= 1:
            raise InvalidParameterError(
                "enough samples to compute standard deviation", f"{n} samples"
            )

        # compute RMSE and standard deviation
        rmse = super()._evaluate(true, pred)
        stddev = np.std(true.mean, ddof=self._bias_correction)

        # ensure sufficient variance in samples
        if stddev <= 1e-3:  # hard-coded, could be initialization parameter
            raise InvalidParameterError(
                "sufficient label variance for non-zero standard deviation",
                f"standard deviation of {stddev}",
            )

        return float(rmse / stddev)


############################
#  Uncertainty statistics  #
############################


class LogPredictiveDensity(VectorEvaluationMetric):
    r"""Logarithmized Predictive Density (LPD)."""

    def _evaluate(self, true, pred):
        r"""Logarithmic Predictive Density (LPD).

        Assumes a normal predictive distribution.

        \[ 
              \log p (y_i = t_i | x_i)
            = - ( \log \sqrt{2\pi} + \log \sigma_i + \frac{1}{2} ( \frac{y_i - t_i}{\sigma_i} )^2 )
        \]

        See, for example, 
            Joaquin Quinonero-Candela, Carl Edward Rasmussen, Fabian Sinz, Olivier Bousquet, and Bernhard Schölkopf. 
            Evaluating predictive uncertainty challenge, p. 1-27, 2005. In Joaquin Quinonero-Candela, Ido Dagan, 
            Bernardo Magnini, and Florence d'Alché Buc (editors), Proceedings of the First PASCAL Machine Learning 
            Challenges Workshop (MLCW 2005), Southampton, United Kingdom, April 11–13, 2005.
        
        Parameters:
            true: observed property distribution; requires only means
            pred: predictive property distribution; must be normal

        Returns:
            logarithmic predictive densities as a NumPy vector of floating point numbers
        """

        true = params.distribution(true)
        pred = params.normal_distribution(pred)
        if np.any(pred.stddev == 0):
            warn(
                f"Some uncertainties are zero. Metric {self.__class__.__name__}"
                "may return nan.",
                RuntimeWarning
            )
        lpd = -(
            np.log(np.sqrt(2 * np.pi))
            + np.log(pred.stddev)
            + 0.5 * np.square((true.mean - pred.mean) / pred.stddev)
        )
        return lpd


class MeanLogPredictiveDensity(ScalarEvaluationMetric):
    """Mean Logarithmized Predictive Density (MLPD)."""

    @property
    def orientation(self):
        """Indicate maximization."""

        return +1

    def _evaluate(self, true, pred):
        r"""Mean Logarithmic Predictive Density (MLPD).

        Mean of LogPredictiveDensity.
        Assumes a normal predictive distribution.

        \[ 
              1/n \sum_{i=1}^n \log p (y_i = t_i | x_i)
            = - ( \log \sqrt{2\pi} + \frac{1}{2n} \sum_{i=1}^n ( \log \sigma_i^2 + \frac{(y_i-t_i)^2}{\sigma_i^2} ) )
        \]

        See LogPredictiveDensity for details.
        
        Parameters:
            true: observed property distribution; requires only means
            pred: predictive property distribution; must be normal

        Returns:
            mean logarithmic predictive densities as a floating point number
        """

        return np.mean(LogPredictiveDensity()._evaluate(true, pred))


class ContinuousRankedProbabilityScore(VectorEvaluationMetric):
    r"""Continuous Ranked Probability Score (CRPS).

    The Continuous Ranked Probability Score (CRPS) [1] is the squared-difference integral 
    between the predicted cumulative distribution function F and that of a delta function 
    on the true value:

	\int\limits_{-\infty}^{\infty} \bigl( F(u) - F_y(u) \bigr)^2 w(u) \mathrm{d} u ,

    where $F_y(u) = 0$ for $u \leq y$ and 1 otherwise, and $w$ is a weighting function.

    For normal predictive distributions, an analytic expression exists: [2]

    \sigma \Bigl( y' \bigl( 2 \Phi(y') - 1 \bigr) + 2 \phi(y') - \frac{1}{\sqrt{\pi}} \Bigr) 
    
    where $y' = \frac{y-\mu}{\sigma}$, and, $\Phi$ and $\phi$ are cumulative and probability 
    density functions of the standard normal distribution.

    [1] James E. Matheson and Robert L. Winkler. Scoring rules for continuous 
        probability distributions. Management Science 22(10):1087–1096, 1976.
    [2] Tilmann Gneiting, Adrian E. Raftery, Anton H. Westveld III, Tom Goldman. Calibrated 
        probabilistic forecasting using ensemble model output statistics and minimum CRPS 
        estimation. Monthly Weather Review, 133(5):1098–1118, 2005.
    """

    def _evaluate(self, true, pred):
        """Evaluate continuous ranked probability score (CRPS).
        
        CRPS depends on the mean of the observations and, in general, the full predictive distribution.

        Currently implemented only for normal predictive distributions, for which a closed-form expression exists. 
        For arbitrary distributions (given as samples), an expression suitable for direct implementation is given by Equ. 3 in

        Eric P. Grimit, Tilmann Gneiting, Veronica J. Berrocal, Nicholas A. Johnson: 
        The continuous ranked probability score for circular variables and its application to mesoscale forecast ensemble verification,
        Quarterly Journal of the Royal Meteorological Society 132(621C): 2925--2942, 2006. DOI 10.1256/qj.05.235

        Parameters:
            true: observed property distributions; requires only means
            pred: predictive property distributions

        Returns:
            sequence of metric values
            continuous ranked probability scores as a NumPy vector of floating point numbers
        """

        true = params.distribution(true)
        pred = params.normal_distribution(pred)
        if np.any(pred.stddev == 0):
            warn(
                f"Some uncertainties are zero. Metric {self.__class__.__name__}"
                "may return nan.",
                RuntimeWarning
            )
        strue = (true.mean - pred.mean) / pred.stddev  # re-used intermediate quantity
        crps = pred.stddev * (
            strue * (2 * sp.stats.norm.cdf(strue) - 1)
            + 2 * sp.stats.norm.pdf(strue)
            - 1 / np.sqrt(np.pi)
        )

        return crps


class MeanContinuousRankedProbabilityScore(ScalarEvaluationMetric):
    """Mean Continuous Ranked Probability Score (mCRPS)."""

    @property
    def orientation(self):
        """Indicate minimization."""

        return -1

    def _evaluate(self, true, pred):
        """Return arithmetic mean of CRPS."""

        return np.mean(ContinuousRankedProbabilityScore()._evaluate(true, pred))


class StandardConfidence(ScalarEvaluationMetric):
    """Fraction of the time that the magnitude of the residual is less than the predicted standard deviation.
    Standard confidence evaluates the quality of the predicted uncertainty estimates, both in terms of individual predictions and overall normalization.
    Does not depend on the predicted values, only the residuals.

    An alternative definition of standard confidence is as the fraction of observations for which the
    "normalized residual" -- residual divided by predicted uncertainty -- is less than one.
    In the ideal case the normalized residuals are normally distributed with std=1, and
    so in the ideal case the standard confidence will be 0.68. Thus there is no "orientation",
    and closer to 0.68 is better.

    The standard confidence is the observed coverage probability at the 68% confidence level.
    See e.g. https://www.stats.ox.ac.uk/pub/bdr/IAUL/Course1Notes5.pdf.
    """

    def _evaluate(self, true, pred):
        """ Compute standard confidence

        Parameters:
            true: observed property distributions; requires only means
            pred: predictive property distributions

        Returns:
            standard confidence
        """

        true = params.distribution(true)
        pred = params.normal_distribution(pred)

        abs_residual = np.abs(true.mean - pred.mean)
        is_less = abs_residual < pred.stddev
        stdconf = np.mean(is_less)

        return stdconf


class RootMeanSquareStandardizedResiduals(ScalarEvaluationMetric):
    """Root Mean Square of the Standardized Residuals (RMSSE).
    
    RMSSE evaluates the quality of the predicted uncertainty estimates, both in terms of individual predictions and overall normalization.
    Compared to standard confidence, RMSSE is more sensitive to outliers.
    Does not depend on the predicted values, only the residuals.

    No "orientation". Closer to 1 is better.
    """

    def _evaluate(self, true, pred):
        """ Compute RMSSE.

        Parameters:
            true: observed property distributions; requires only means
            pred: predictive property distributions

        Returns:
            RMSSE
        """

        true = params.distribution(true)
        pred = params.normal_distribution(pred)
        if np.any(pred.stddev == 0):
            warn(
                f"Some uncertainties are zero. Metric {self.__class__.__name__}"
                "will be nan.",
                RuntimeWarning
            )
            return np.nan
        strue = (true.mean - pred.mean) / pred.stddev
        rmsse = np.sqrt(np.mean(np.power(strue, 2)))

        return rmsse


class UncertaintyCorrelation(ScalarEvaluationMetric):
    """Correlation between uncertainty estimate and abs(residual). 
    A positive value is desirable. A negative value indicates pathological behavior.
    Does not depend on the predicted values, only the residuals.
    """

    @property
    def orientation(self):
        """Indicate maximization."""

        return +1

    def _evaluate(self, true, pred):
        """Compute Uncertainty Correlation

        Parameters:
            true: observed property distributions; requires only means
            pred: predictive property distributions

        Returns:
            uncertainty correlation
        """

        true = params.distribution(true)
        pred = params.normal_distribution(pred)

        abs_residual = np.abs(true.mean - pred.mean)
        uc_corr = np.corrcoef(abs_residual, pred.stddev)[0,1] # get off-diagonal of correlation matrix

        return uc_corr


# helper function 

def two_sample_cumulative_distribution_function_statistic(
    sample_a, sample_b, f=lambda p, t: np.square(p - t), g=lambda s, w: np.sum(s * w)
):
    r"""Compute a statistic of the difference between two empirical cumulative distribution functions.
    
    Calculate statistics of the cumulative distribution functions (CDF) of two samples.
    Let $x_1,\ldots,x_d$ be the union of the two samples, $x_i < x_{i+1}$, and let 
    $w_i = x_{i+1}-x_i$, $i = 1,\ldots,d-1$ be the differences between them.
    The calculated statistics have the form $g(s,w)$ where $s_i = f(F_a(x_i), F_b(x_i))$)
    and $F_a$, $F_b$ are the CDFs of the two samples.

    Here, the $x_i$ are the points where one or both of the CDFs changes, $f$ is a statistic
    that depends on the value of the two CDFs, and $g$ is an arbitrary function of $s$ and $w$.
    
    The default choice for $g$ is Riemann integration; as the CDFs are step functions, this is exact
    and leads to statistics of the form
    
    \[ \int_{-\infty}^{\infty} f(F_a(x),F_b(x)) dx . \]

    Parameters:
        sample_a: first sample; a sequence of real numbers
        sample_b: second sample; a sequence of real numbers; 
                  can be of different length than first sample
        f: function accepting two same-length real vectors, returning a real vector of same length. 
           This function computes a value that depends only on the two CDFs, and is thus constant 
           between change points. The default is the squared difference, f(a,b) = np.square(a-b).
           The convention here is to use the left endpoint of the "steps".
        g: function accepting two same-length real vectors, returning a real number. 
           Computes the statistic based on values of f and step "widths". 
           The default, g(s,w) = np.sum(g * w), performs Riemann integration. 
    """

    sample_a = params.real_vector(sample_a)
    sample_b = params.real_vector(sample_b)

    allx = np.union1d(sample_a, sample_b)  # all x where F_a and F_b change
    xdif = np.ediff1d(allx)  # width of Riemann integration bars
    allx = allx.reshape((len(allx), 1))
    cdfa = np.count_nonzero(np.sort(sample_a) <= allx, axis=1) / len(sample_a)
    cdfb = np.count_nonzero(np.sort(sample_b) <= allx, axis=1) / len(sample_b)
    stat = np.asfarray(f(cdfa, cdfb))

    return g(stat[:-1], xdif)
