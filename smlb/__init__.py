"""Package initialization.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2019-2020, Matthias Rupp, Citrine Informatics.
"""

from .exceptions import BenchmarkError, InvalidParameterError
from .object import SmlbObject
from .utility import is_sequence, which
from .parameters import params
from .random import Random
from .physchem import element_data
from .data import Data, intersection, complement
from .tabular_data import TabularData, TabularDataFromPandas
from .vector_space_data import VectorSpaceData
from .noise import Noise, NoNoise, NormalNoise, LabelNoise
from .transformations import (
    DataTransformation,
    DataValuedTransformation,
    IdentityTransformation,
    InvertibleTransformation,
    DataTransformationFailureMode
)
from .features import Features, IdentityFeatures
from .learners import Learner, UnsupervisedLearner, SupervisedLearner
from .sampling import Sampler, RandomSubsetSampler, RandomVectorSampler, GridSampler
from .metrics import (
    EvaluationMetric,
    ScalarEvaluationMetric,
    Residuals,
    AbsoluteResiduals,
    MeanAbsoluteError,
    SquaredResiduals,
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
    two_sample_cumulative_distribution_function_statistic,
)
from .distributions import (
    PredictiveDistribution,
    DeltaPredictiveDistribution,
    NormalPredictiveDistribution,
    CorrelatedNormalPredictiveDistribution,
)
from .scorer import Scorer, LIScorer
from .optimizer import Optimizer, TrackedTransformation
from .evaluations import Evaluation, EvaluationConfiguration
from .plots import Plot, PlotConfiguration, GeneralizedFunctionPlot, LearningCurvePlot
from .workflow import Workflow
from .java import JavaGateway
