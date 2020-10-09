"""WatercolorPigments2019Data tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2020, Matthias Rupp, Citrine Informatics.
"""

from smlb.datasets.experimental.watercolor_pigments_c19.watercolor_pigments_c19 import (
    WatercolorPigments2019Dataset,
    WatercolorPigments2019DatasetFeatures,
)


###################################
#  WatercolorPigments2019Dataset  #
###################################


def test_watercolors_pigments_2019_1():
    """Simple checks against the original dataset."""

    # test correctness of random entries
    # assumes that the order within primary and 1:1, 1:2, 2:1 mixtures remains unchanged

    # primary pigments
    wcp = WatercolorPigments2019Dataset(filter_="primary")
    e, y = wcp.samples([0])[0], wcp.labels([0])[0]
    assert e["identifier"] == "C1_R_0.01ml" and e["index"] == 1 and all(y == (218, 55, 50))
    e, y = wcp.samples([104])[0], wcp.labels([104])[0]  # line 106 in input file
    assert e["identifier"] == "C15_R_0.09ml" and e["index"] == 9 and all(y == (41, 75, 175))

    # 1:1 mixtures
    wcm = WatercolorPigments2019Dataset(filter_=lambda e: e["type"] == "mixture11")
    e, y = wcm.samples([2])[0], wcm.labels([2])[0]
    assert (
        e["identifier"] == "R_C1C6_1_1"
        and e["indexA"] == 1
        and e["indexB"] == 4
        and e["concentrationA"] == 0.01
        and e["concentrationB"] == 0.01
        and all(y == (218, 63, 50))
    )

    # 1:2 mixtures
    wcm = WatercolorPigments2019Dataset(filter_=lambda e: e["type"] == "mixture12")
    e, y = wcm.samples([193])[0], wcm.labels([193])[0]
    assert (
        e["identifier"] == "R_C16C21_4_8"
        and e["indexA"] == 10
        and e["indexB"] == 12
        and e["concentrationA"] == 0.04
        and e["concentrationB"] == 0.08
        and all(y == (47, 47, 50))
    )

    # 2:1 mixtures
    wcm = WatercolorPigments2019Dataset(filter_=lambda e: e["type"] == "mixture21")
    e, y = wcm.samples([89])[0], wcm.labels([89])[0]
    assert (
        e["identifier"] == "R_C8C14_4_2"
        and e["indexA"] == 6
        and e["indexB"] == 8
        and e["concentrationA"] == 0.04
        and e["concentrationB"] == 0.02
        and all(y == (61, 74, 62))
    )


###########################################
#  WatercolorPigments2019DatasetFeatures  #
###########################################


def test_watercolors_pigments_2019_features_1():
    """Test one-hot encoding"""

    # no primary pigments
    wcm = WatercolorPigments2019Dataset(filter_="mixture")
    wcmf = WatercolorPigments2019DatasetFeatures(encoding="one-hot").fit(wcm).apply(wcm)

    assert wcmf.samples().shape == (256 + 198 + 190, 2 * 13)
    assert (
        wcmf.samples([256])[0]
        == [0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ).all()
