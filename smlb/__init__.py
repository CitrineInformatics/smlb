"""Package initialization.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2019-2020, Matthias Rupp, Citrine Informatics.
"""

from smlb.core.exceptions import BenchmarkError, InvalidParameterError
from smlb.core.object import SmlbObject
from smlb.core.utility import is_sequence, which
from smlb.core.parameters import params
from smlb.core.random import Random
from smlb.core.physchem import element_data
from smlb.core.data import Data, intersection, complement
from smlb.core.tabular_data import TabularData, TabularDataFromPandas
from smlb.core.vector_space_data import VectorSpaceData
from smlb.core.transformations import (
    DataTransformation,
    DataValuedTransformation,
    IdentityTransformation,
    InvertibleTransformation,
    DataTransformationFailureMode,
)
from smlb.core.features import Features, IdentityFeatures
from smlb.core.sampling import Sampler, RandomSubsetSampler, RandomVectorSampler, GridSampler
from smlb.core.distributions import (
    PredictiveDistribution,
    DeltaPredictiveDistribution,
    NormalPredictiveDistribution,
    CorrelatedNormalPredictiveDistribution,
)
from smlb.core.evaluations import Evaluation, EvaluationConfiguration
from smlb.core.plots import Plot, PlotConfiguration, GeneralizedFunctionPlot, LearningCurvePlot
from smlb.core.workflow import Workflow
from smlb.core.noise import Noise, NoNoise, NormalNoise, LabelNoise
from smlb.core.learners import Learner, UnsupervisedLearner, SupervisedLearner
from smlb.core.metrics import (
    ScalarEvaluationMetric,
    VectorEvaluationMetric,
    Residuals,
    AbsoluteResiduals,
    SquaredResiduals,
    MeanAbsoluteError,
    MeanSquaredError,
    RootMeanSquaredError,
    StandardizedRootMeanSquaredError,
    LogPredictiveDensity,
    MeanLogPredictiveDensity,
    ContinuousRankedProbabilityScore,
    MeanContinuousRankedProbabilityScore,
    StandardConfidence,
    RootMeanSquareStandardizedResiduals,
    UncertaintyCorrelation,
)
