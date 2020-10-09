"""ChemistryDevelopmentKitMoleculeFeatures tests.

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
2019-2020, Citrine Informatics.
"""

import hashlib
import os
import urllib

import numpy as np

import pytest

import smlb

from smlb.features.chemistry_development_kit_molecules import (
    ChemistryDevelopmentKitMoleculeFeatures,
    CdkJavaGateway,
)

# access to CDK .jar
_cdk_jar_filename = os.path.join("build", "cdk.jar")
if not os.access(_cdk_jar_filename, os.R_OK):
    # automatically download CDK jar
    # https://github.com/cdk/cdk/releases/latest
    # verify by checksum for security
    cdk_version = "cdk-2.3"
    cdk_jar_path = os.path.join("build", "cdk.jar")
    cdk_hash = "bf60002d40b88136f185a96b1f38135e240733360c6b684cfc2bfd64972eb04a"
    try:
        with urllib.request.urlopen(
            f"https://github.com/cdk/cdk/releases/download/{cdk_version}/{cdk_version}.jar"
        ) as cdk_jar:
            cdk_jar_contents = cdk_jar.read()
            hash_actual = hashlib.sha256(cdk_jar_contents).hexdigest()
            hash_expected = cdk_hash
            assert hash_actual == hash_expected, "CDK jar checksum is invalid. Aborting."
            with open(cdk_jar_path, "wb") as f:
                f.write(cdk_jar_contents)
    except AssertionError:
        pytest.skip(
            "Downloaded CDK .jar file has wrong checksum. "
            "Please download from 'https://github.com/cdk/cdk/releases/latest' as 'build/cdk.jar'",
            allow_module_level=True,
        )
    except Exception:
        pytest.skip(
            "Could not access CDK .jar file. "
            "Please download from 'https://github.com/cdk/cdk/releases/latest' as 'build/cdk.jar'",
            allow_module_level=True,
        )


@pytest.mark.timeout(5)
def test_ChemistryDevelopmentKitMoleculeFeatures_1():
    """Simple examples."""

    # specific descriptors

    # citric acid, three carboxylic groups
    data = smlb.TabularData(data=np.array(["OC(=O)CC(O)(C(=O)O)CC(=O)O"]))
    features = (
        ChemistryDevelopmentKitMoleculeFeatures(
            # using an order different from the order in which descriptors are defined
            # in ChemistryDevelopmentKitMoleculeFeatures to test that descriptors are
            # calculated in the order specified by `select`
            select=["acidic_group_count", "bond_count", "atom_count"],
        )
        .fit(data)
        .apply(data)
    )
    assert features.samples()[0][0] == 3
    assert features.samples()[0][1] == 12
    assert features.samples()[0][2] == 21

    # all descriptors

    # citric acid, benzene
    data = smlb.TabularData(data=np.array(["OC(=O)CC(O)(C(=O)O)CC(=O)O", "c1ccccc1"]))
    features = (
        ChemistryDevelopmentKitMoleculeFeatures(
            select=ChemistryDevelopmentKitMoleculeFeatures.PRESET_ALL,
        )
        .fit(data)
        .apply(data)
    )
    dimensions = (v[1] for v in ChemistryDevelopmentKitMoleculeFeatures.DESCRIPTORS.values())
    assert len(features.samples()[0]) == sum(dimensions)

    # pre-sets
    features = (
        (
            ChemistryDevelopmentKitMoleculeFeatures(
                select=ChemistryDevelopmentKitMoleculeFeatures.PRESET_ROBUST,
            )
        )
        .fit(data)
        .apply(data)
    )

    # fragile descriptors
    data = smlb.TabularData(data=np.array(["CCCCl"]))
    features = (
        (ChemistryDevelopmentKitMoleculeFeatures(select=["alogp"])).fit(data).apply(data)
    ).samples()[0]
    assert np.allclose((features[0], features[2]), (1.719, 20.585), atol=0.01)

    # raise for unknown descriptors
    with pytest.raises(smlb.InvalidParameterError):
        ChemistryDevelopmentKitMoleculeFeatures(select=["atoms_counts"])

    # raise for invalid cdk_path
    with pytest.raises(smlb.InvalidParameterError):
        ChemistryDevelopmentKitMoleculeFeatures(
            CdkJavaGateway(cdk_jar_path="/nonexisting/path/to/cdk.jar")
        )

    # todo: this is a temporary fix for problems in the interaction between
    #        ChemistryDevelopmentKitMoleculeFeatures and lolopy. If the
    #        JavaGateway for CDK is not shut down, lolopy hangs on querying
    #        the port number of its server:
    #        ../../../virtualenv/python3.6.7/lib/python3.6/site-packages/lolopy/loloserver.py:74: in get_java_gateway
    #        >       _port = int(proc.stdout.readline())
    #        E       Failed: Timeout >10.0s
    #        ../../../virtualenv/python3.6.7/lib/python3.6/site-packages/py4j/java_gateway.py:332: Failed
    CdkJavaGateway()._shutdown_gateway()


@pytest.mark.timeout(5)
def test_ChemistryDevelopmentKitMoleculeFeatures_2():
    """Failures during SMILES parsing."""

    # specific descriptors

    # "raise"
    data = smlb.TabularData(data=np.array(["[NH]c1cc[nH]nn1"]))
    features = ChemistryDevelopmentKitMoleculeFeatures(select=["atom_count"], failmode="raise")
    with pytest.raises(smlb.BenchmarkError):
        features.fit(data).apply(data)

    # "drop"
    data = smlb.TabularData(data=np.array(["N", "[NH]c1cc[nH]nn1", "O"]))
    features = ChemistryDevelopmentKitMoleculeFeatures(select=["atom_count"], failmode="drop")
    data = features.fit(data).apply(data)
    assert (data.samples() == [[4], [3]]).all()

    # "mask"
    data = smlb.TabularData(data=np.array(["N", "[NH]c1cc[nH]nn1", "O"]))
    mask = np.empty(3, dtype=bool)
    features = ChemistryDevelopmentKitMoleculeFeatures(
        select=["atom_count"], failmode=("mask", mask)
    )
    data = features.fit(data).apply(data)
    assert (mask == [False, True, False]).all()

    # "index"
    data = smlb.TabularData(data=np.array(["N", "[NH]c1cc[nH]nn1", "O"]))
    index = []
    features = ChemistryDevelopmentKitMoleculeFeatures(
        select=["atom_count"], failmode=("index", index)
    )
    data = features.fit(data).apply(data)
    assert index == [1]

    # todo: this is a temporary fix for problems in the interaction between
    #        ChemistryDevelopmentKitMoleculeFeatures and lolopy. If the
    #        JavaGateway for CDK is not shut down, lolopy hangs on querying
    #        the port number of its server:
    #        ../../../virtualenv/python3.6.7/lib/python3.6/site-packages/lolopy/loloserver.py:74: in get_java_gateway
    #        >       _port = int(proc.stdout.readline())
    #        E       Failed: Timeout >10.0s
    #        ../../../virtualenv/python3.6.7/lib/python3.6/site-packages/py4j/java_gateway.py:332: Failed
    CdkJavaGateway()._shutdown_gateway()
