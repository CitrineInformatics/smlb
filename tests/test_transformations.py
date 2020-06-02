"""DataTransformation tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
"""

import pytest

import numpy as np

import smlb

###########################
#  InverseTransformation  #
###########################


@pytest.fixture
def fixture_VectorSpaceData_parabola_1d():
    f = lambda v: v[0] ** 2
    ds = smlb.VectorSpaceData(dimensions=1, function=f, domain=[-2, 2])
    return ds


def test_InverseTransformation():
    """Simple example."""

    class TestInverseTransformation(smlb.DataValuedTransformation):
        """Transforms strings back to integers."""

        def fit(self, data):
            return self

        def apply(self, data):
            return smlb.TabularData(data=np.array([[int(i)] for i in data]))

    class TestTransformation(smlb.DataTransformation, smlb.InvertibleTransformation):
        """Transforms integers to strings."""

        def fit(self, data):
            return self

        def apply(self, data):
            return [str(i[0]) for i in data.samples()]

        def inverse(self):
            return TestInverseTransformation()

    original_data = smlb.TabularData(np.array([[1], [2], [3], [5], [8]]))
    transformed_data = TestTransformation().fit(original_data).apply(original_data)
    assert transformed_data == ["1", "2", "3", "5", "8"]
    preimage_data = (
        TestTransformation().fit(original_data).inverse().apply(transformed_data)
    )  # no need to fit
    assert all(preimage_data.samples() == original_data.samples())


###################################
#  DataTransformationFailureMode  #
###################################


def test_DataTransformationFailureMode_no_duplicates():
    """Test that only unique indices are returned."""

    dataset = smlb.TabularData(data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    fails = []
    failmode = smlb.DataTransformationFailureMode(("index", fails), dataset.num_samples)
    failmode.handle_failure(1)
    failmode.handle_failure(5)
    failmode.handle_failure(6)
    failmode.handle_failure(5)
    dataset = failmode.finalize(dataset)

    assert dataset.num_samples == 10
    assert fails == [1, 5, 6]
