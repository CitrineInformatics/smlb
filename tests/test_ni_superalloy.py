"""Ni superalloys dataset tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2019, Brendan Folie, Citrine Informatics.
"""

import numpy as np

from smlb.datasets.experimental.ni_superalloys.Ni_superalloy import NiSuperalloyDataset


def test_ni_superalloy_basic():
    """Instantiate and check basic properties of a Ni Superalloy Dataset object."""

    dataset = NiSuperalloyDataset()
    data = dataset.samples()
    labels = dataset.labels()

    # data should be a numpy array
    assert isinstance(data, np.ndarray)
    # Every entry should be a string or a number that is not NaN
    for row in data:
        for entry in row:
            assert isinstance(entry, str) or not np.isnan(entry)

    # labels should be a numpy array
    assert isinstance(labels, np.ndarray)

    # data and labels should have the same number of rows
    assert dataset.num_samples == len(dataset.labels())

    # each entry of labels should be a float (nan is allowed)
    for label in labels.reshape(-1):
        assert isinstance(label, float)

    # ignoring dubious samples should result in fewer samples (but more than 0)
    dataset_restricted = NiSuperalloyDataset(ignore_dubious=True)
    assert 0 < dataset_restricted.num_samples < dataset.num_samples


def test_ni_superalloy_limited_labels():
    """Instantiate with a restricted set of labels."""

    labels_to_load = ["Elongation", "Stress Rupture Time"]
    dataset = NiSuperalloyDataset(labels_to_load=labels_to_load)
    assert dataset.labels().shape[1] == len(labels_to_load)
