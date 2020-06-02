"""Qm9RamakrishnanEtAl2014Dataset tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp, 2020.
"""

import os.path

import numpy as np
import pytest

pytest.importorskip("datasets.experimental.qm9_ramakrishnan_etal14")
from datasets.experimental.qm9_ramakrishnan_etal14.qm9_ramakrishnan_etal14 import (
    Qm9RamakrishnanEtAl2014Dataset,
)  # noqa E402

from test_data import validate_data_interface

# test dataset
# a small subset of the whole database.
# contains molecules with indices 1-7 (regular), 58, 21490 (uncharacterized), 6620 (unconverged)
_qm9_filename = os.path.join(os.path.dirname(__file__), "test_qm9_ramakrishnan_etal14_dataset.zip")


def test_qm9_ramakrishnan_etal14_dataset_init():
    """Tests instantiating dataset."""

    # with and without uncharacterized and unconverged molecules
    ds = Qm9RamakrishnanEtAl2014Dataset(source=_qm9_filename)
    assert ds.num_samples == 10

    ds = Qm9RamakrishnanEtAl2014Dataset(source=_qm9_filename, exclude_uncharacterized=True)
    assert ds.num_samples == 8

    ds = Qm9RamakrishnanEtAl2014Dataset(source=_qm9_filename, exclude_unconverged=True)
    assert ds.num_samples == 9

    ds = Qm9RamakrishnanEtAl2014Dataset(
        source=_qm9_filename, exclude_uncharacterized=True, exclude_unconverged=True
    )
    assert ds.num_samples == 7

    # [join], filterf, samplef, labelf
    filterf = lambda arg: len(arg["atomic_number"]) > 5  # noqa E731
    ds = Qm9RamakrishnanEtAl2014Dataset(source=_qm9_filename, filterf=filterf)
    assert ds.num_samples == 4

    samplef = lambda arg: arg["smiles_gdb9"]  # noqa E731
    ds = Qm9RamakrishnanEtAl2014Dataset(source=_qm9_filename, samplef=samplef)
    assert ds.samples()[-1][0] == "CC#CC#CC(C)=O"

    labelf = lambda arg: arg["C"]  # noqa E731
    ds = Qm9RamakrishnanEtAl2014Dataset(source=_qm9_filename, labelf=labelf)
    assert ds.labels()[0][0] == 157.70699


def test_qm9_ramakrishnan_etal14_dataset_values():
    """Tests retrieving correct values from dataset."""

    ds = Qm9RamakrishnanEtAl2014Dataset(source=_qm9_filename)

    # 3
    # gdb 3	799.58812	437.90386	282.94545	1.8511	6.31	-0.2928	0.0687	0.3615	19.0002	0.021375	-76.404702	-76.401867	-76.400922	-76.422349	6.002
    # O	-0.0343604951	 0.9775395708	 0.0076015923	-0.589706
    # H	 0.0647664923	 0.0205721989	 0.0015346341	 0.294853
    # H	 0.8717903737	 1.3007924048	 0.0006931336	 0.294853
    # 1671.4222	3803.6305	3907.698
    # O	O
    # InChI=1S/H2O/h1H2	InChI=1S/H2O/h1H2

    s = ds.samples([2])[0]
    assert s["index"] == 3
    assert (s["atomic_number"] == np.array([8, 1, 1])).all()
    assert np.allclose(
        s["coordinates"],
        [
            [-0.0343604951, 0.9775395708, 0.0076015923],
            [0.0647664923, 0.0205721989, 0.0015346341],
            [0.8717903737, 1.3007924048, 0.0006931336],
        ],
    )
    assert np.allclose(s["mulliken_charges"], [-0.589706, 0.294853, 0.294853])
    assert np.allclose(s["frequencies"], [1671.4222, 3803.6305, 3907.698])
    assert s["smiles_gdb9"] == s["smiles_relaxed"] == "O"
    assert s["inchi_gdb9"] == s["inchi_relaxed"] == "InChI=1S/H2O/h1H2"

    lbl = ds.labels([2])[0]
    assert np.allclose(
        [
            lbl[ln]
            for ln in [
                "A",
                "B",
                "C",
                "mu",
                "alpha",
                "homo",
                "lumo",
                "gap",
                "r2",
                "zpve",
                "U0",
                "U",
                "H",
                "G",
                "Cv",
            ]
        ],
        [
            799.58812,
            437.90386,
            282.94545,
            1.8511,
            6.31,
            -0.2928,
            0.0687,
            0.3615,
            19.0002,
            0.021375,
            -76.404702,
            -76.401867,
            -76.400922,
            -76.422349,
            6.002,
        ],
    )


def test_qm9_ramakrishnan_etal14_dataset_other():
    """Tests various aspects of dataset."""

    # ds = Qm9RamakrishnanEtAl2014Dataset(source=_qm9_filename)
    # set subset, intersection, complement will fail because of inability to compare (for sorting)
    # the rows of the dataset.
    # assert validate_data_interface(ds)
