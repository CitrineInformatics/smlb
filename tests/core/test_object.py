"""Object base class tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.
"""

import pytest

import numpy as np
import numpy.testing

import smlb


######################
#  Unused arguments  #
######################


def test_smlbobject_unused_initalizer_arguments():
    """Tests the unused-initializer-parameters functionality provided by SmlbObject."""

    # diamond inheritance example
    with pytest.raises(smlb.InvalidParameterError):

        class A(smlb.SmlbObject):
            pass

        class B(smlb.SmlbObject):
            pass

        class C(A, B):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        C(None)

    # example from distributions
    with pytest.raises(smlb.InvalidParameterError):
        smlb.DeltaPredictiveDistribution([1, 2, 3], invalid=None)

    # example from evaluation metrics
    with pytest.raises(smlb.InvalidParameterError):
        smlb.MeanContinuousRankedProbabilityScore(hellohello=99)
