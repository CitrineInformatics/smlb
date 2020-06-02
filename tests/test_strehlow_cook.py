"""Band gaps from Strehlow & Cook 1973 dataset tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.
"""

import pytest

import numpy as np

import smlb


def test_band_gaps_strehlow_cook():
    """Tests instantiating Strehlow & Cook band gaps dataset."""

    from datasets.experimental.band_gaps_sc73.band_gaps_sc73 import BandGapsStrehlowCook1973Dataset

    bg = BandGapsStrehlowCook1973Dataset(filter_="bg")

    assert len(bg.samples()) == 1447
    assert (
        len([e for e in bg.samples() if e["temperature"] is not None]) == 1332
    )  # temperature information

    # when filtering by temperature, all entries have temperature in the right range
    bg = BandGapsStrehlowCook1973Dataset(filter_="t300pm10")
    for s in bg.samples():
        assert 290 <= s["temperature"] <= 310

    # when filtering by temperature and crystallinity, all entries obey these constraints
    bg = BandGapsStrehlowCook1973Dataset(filter_="t300pm10_mc")
    for s in bg.samples():
        assert (
            290 <= s["temperature"] <= 310
            and s["crystallinity"] == BandGapsStrehlowCook1973Dataset.CRYSTALLINITY_SINGLE
        )

    # mis-specified filter raises
    with pytest.raises(smlb.InvalidParameterError):
        BandGapsStrehlowCook1973Dataset(filter_="t300pm11")

    # integer join argument yields correct minimal group size
    bg = BandGapsStrehlowCook1973Dataset(join=4)  # implicit filter_='bg' argument
    for p in bg.labels():
        assert len(p) >= 4

    # combination of filter and join, default value for join
    bg = BandGapsStrehlowCook1973Dataset(filter_="t300pm10_mc", join=True)
    for (s, p) in zip(bg.samples(), bg.labels()):
        assert len(s["crystallinity"]) == len(p) and len(p) >= 1
