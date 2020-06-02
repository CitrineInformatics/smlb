"""MatminerCompositionFeatures tests.

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
2019-2020, Citrine Informatics.
"""

import pytest

import numpy as np

matminer = pytest.importorskip("matminer")
pymatgen = pytest.importorskip("pymatgen")

import smlb

from features.matminer_composition import MatminerCompositionFeatures


def test_MatminerCompositionFeatures_1():
    """Simple examples."""

    # callable, without labels
    data = smlb.TabularData(data=np.array(["LiF", "Sb2Te3"]))
    feat = MatminerCompositionFeatures().fit(data=data).apply(data=data)

    assert isinstance(feat, smlb.TabularData)
    assert feat.is_finite and not feat.is_labeled
    smlb.params.real_matrix(feat.samples())  # must not raise

    # callable, with labels
    data = smlb.TabularData(data=np.array(["LiF", "Sb2Te3"]), labels=np.array([1.0, 2.0]))
    feat = MatminerCompositionFeatures().fit(data=data).apply(data=data)

    assert isinstance(feat, smlb.TabularData)
    smlb.params.real_matrix(feat.samples())  # must not raise
    smlb.params.real_vector(feat.labels())  # must not raise

    # third example
    data = smlb.TabularData(data=np.array(["Al2O3", "Ni1.8W.05Al0.4"]))
    feat = MatminerCompositionFeatures(ionic_fast=True).fit(data=data).apply(data=data)

    assert isinstance(feat, smlb.TabularData)
    smlb.params.real_matrix(feat.samples())  # must not raise


def test_MatminerCompositionFeatures_2():
    """Test that feature subsets can be applied individually."""

    data = smlb.TabularData(data=np.array(["V4O5", "Ni87.3Al10Cu3.3Co.23"]))
    mmfa = (
        MatminerCompositionFeatures(select="all", ionic_fast=True).fit(data=data).apply(data=data)
    )
    mmfb = (
        MatminerCompositionFeatures(
            select=("stoichiometry", "elemental", "ionic", "valence"), ionic_fast=True
        )
        .fit(data=data)
        .apply(data=data)
    )

    mmf1 = MatminerCompositionFeatures(select="stoichiometry").fit(data=data).apply(data=data)
    mmf2 = MatminerCompositionFeatures(select="elemental").fit(data=data).apply(data=data)
    mmf3 = (
        MatminerCompositionFeatures(select="ionic", ionic_fast=True)
        .fit(data=data)
        .apply(data=data)
    )
    mmf4 = MatminerCompositionFeatures(select="valence").fit(data=data).apply(data=data)

    # stack the individual featurizations together and assert that we recover full featurization
    recombined_features = np.hstack(
        [mmf1.samples(), mmf2.samples(), mmf3.samples(), mmf4.samples()]
    )

    assert (recombined_features == mmfa.samples()).all()
    assert (mmfa.samples() == mmfb.samples()).all()


def test_MatminerCompositionFeatures_3():
    """Test specific values for each feature group.

    These tests compute all wrapped feature groups
    (e.g., stoichiometry, elemental, ionic, valence)
    for reference systems (e.g., Fe2 O3) and compare
    against reference values provided by the matminer tests.

    Reference values from matminer `test_composition.py`:
    https://github.com/hackingmaterials/matminer/blob/master/matminer/featurizers/tests/test_composition.py

    The tests proceed according to this scheme:
    ```
        # create an (unlabeled) dataset containing one or two chemical sum formulas
        data = smlb.TabularData(data=["compound(s) formula", ...])
        # compute a specific group of matminer features; some accept parameters passed through to matminer
        mmf = MatminerCompositionFeatures(select="group", pass-through parameters)
        # compute the features; mmf is now a dataset that contains feature vectors
        mmf = mmf.fit(data).apply(data)
        # compare the i-th feature of first sample versus matminer reference value
        assert np.allclose(mmf.samples()[0][i], reference_value)
    ```
    The reference values are taken from matminer. They do not have any meaning beyond
    having been computed there. This test only verifies that the smlb wrapper returns
    the same values as the original matminer call for selected test cases.
    """

    # stoichiometry

    # default
    data = smlb.TabularData(data=np.array(["Fe2O3"]))
    mmf = MatminerCompositionFeatures(select="stoichiometry").fit(data).apply(data)
    assert mmf.samples()[0][0] == 2
    assert np.allclose(mmf.samples()[0][-2], 0.604895199)

    # user-defined norms
    mmf = (
        MatminerCompositionFeatures(select="stoichiometry", stoichiometry_p_list=(7, 0))
        .fit(data)
        .apply(data)
    )
    assert np.allclose(mmf.samples()[0][0], 0.604895199)
    assert mmf.samples()[0][1] == 2

    # invariance to amounts
    data = smlb.TabularData(np.array(["FeO", "Fe0.5O0.5", "Fe2O2"]))
    mmf = MatminerCompositionFeatures(select="stoichiometry").fit(data).apply(data)
    assert np.allclose(mmf.samples()[0], mmf.samples()[1])
    assert np.allclose(mmf.samples()[0], mmf.samples()[2])

    # elemental

    # magpie
    data = smlb.TabularData(np.array(["Fe2O3"]))
    mmf = MatminerCompositionFeatures(select="elemental").fit(data).apply(data)
    assert np.allclose(mmf.samples()[0][:6], [8, 26, 18, 15.2, 8.64, 8])

    # ionic

    # default
    data = smlb.TabularData(data=np.array(["Fe2O3"]))
    mmf = MatminerCompositionFeatures(select="ionic").fit(data=data).apply(data=data)
    assert np.allclose(mmf.samples()[0], [1, 0.476922164, 0.114461319])

    # fast
    mmf = (
        MatminerCompositionFeatures(select="ionic", ionic_fast=True)
        .fit(data=data)
        .apply(data=data)
    )
    assert np.allclose(mmf.samples()[0], [1, 0.476922164, 0.114461319])

    # fast with heterovalent compound
    data = smlb.TabularData(data=np.array(["Fe3O4"]))
    mmf1 = MatminerCompositionFeatures(select="ionic", ionic_fast=False).fit(data).apply(data)
    mmf2 = MatminerCompositionFeatures(select="ionic", ionic_fast=True).fit(data).apply(data)
    assert mmf1.samples()[0][0] == 1 and mmf2.samples()[0][0] == 0

    # valence

    # default parameters
    data = smlb.TabularData(np.array(["Fe2O3"]))
    mmf = MatminerCompositionFeatures(select="valence").fit(data).apply(data)
    np.allclose(mmf.samples()[0], [2.0, 2.4, 2.4, 0.0, 0.294117647, 0.352941176, 0.352941176, 0])

    # user-defined parameters
    data = smlb.TabularData(np.array(["Fe2O3"]))
    mmf = (
        MatminerCompositionFeatures(
            select="valence", valence_orbitals=("s", "p"), valence_props=("avg",)
        )
        .fit(data)
        .apply(data)
    )
    np.allclose(mmf.samples()[0], [2.0, 2.4])

    data = smlb.TabularData(np.array(["Fe2O3"]))
    mmf = (
        MatminerCompositionFeatures(
            select="valence", valence_orbitals=("p", "s"), valence_props=("frac", "avg",)
        )
        .fit(data)
        .apply(data)
    )
    np.allclose(mmf.samples()[0], [0.352941176, 0.294117647, 2.4, 2.0])
