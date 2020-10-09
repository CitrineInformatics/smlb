"""SuperconductorsCitrine2016Dataset tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2020 Citrine Informatics.
"""

import numpy as np

import smlb


def test_superconductors_citrine16_1():
    """Tests instantiating dataset."""

    from smlb.datasets.experimental.superconductors_citrine16.superconductors_citrine16 import (
        SuperconductorsCitrine2016Dataset,
    )

    # all data
    sc = SuperconductorsCitrine2016Dataset(process=False, join=False)
    assert sc.num_samples == 588

    sc = SuperconductorsCitrine2016Dataset(process=True, join=False)
    assert np.max(sc.labels()) == 134  # from LOCO-CV paper

    # filter some
    sc = SuperconductorsCitrine2016Dataset(
        process=True, join=False, filter_=lambda e: not e["flagged_formula"]
    )
    assert sc.num_samples < 588

    # filter some more
    n1 = sc.num_samples
    sc = SuperconductorsCitrine2016Dataset(
        process=True,
        join=False,
        filter_=lambda e: not e["flagged_formula"] and not e["process_Tc/K"],
    )
    assert sc.num_samples < n1

    # conservative parametrization from docstring
    n2 = sc.num_samples
    sc = SuperconductorsCitrine2016Dataset(
        process=True,
        join=True,
        filter_=lambda e: not any(e["flagged_formula"]),
        samplef=lambda e: e["formula"],
        labelf=lambda tc: np.median(tc),
    )
    assert 0 < sc.num_samples < n2
    assert smlb.params.tuple_(sc.samples(), smlb.params.string)
    assert smlb.params.tuple_(sc.labels(), smlb.params.real)
