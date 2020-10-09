"""SuperconductorsCitrine2016Dataset tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2020 Citrine Informatics.
"""

import io

import numpy as np

# numbers are not actual numbers from dataset
example_input = """ "id", "SMILES_str", "stoich_str", "mass", "pce", "voc", "jsc", "e_homo_alpha", "e_gap_alpha", "e_lumo_alpha", "tmp_smiles_str"
"655365" , "C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1"               , "C18H9N3OSSe", "39.3", "5.1", "0.86", "91.5", "-5.4", "2.0", "-3.4", "C1=CC=C(C1)c1cc2[se]c3c4occc4c4nsnc4c3c2cn1"
"1245190", "C1C=CC=C1c1cc2[se]c3c(ncc4ccccc34)c2c2=C[SiH2]C=c12"     , "C22H15NSeSi", "40.4", "5.2", "0.50", "16.4", "-5.1", "1.6", "-3.4", "C1=CC=C(C1)c1cc2[se]c3c(ncc4ccccc34)c2c2=C[SiH2]C=c12"
"21847"  , "C1C=c2ccc3c4c[nH]cc4c4c5[SiH2]C(=Cc5oc4c3c2=C1)C1=CC=CC1", "C24H17NOSi" , "36.4", "0"  , "0"   , "19.4", "-4.5", "1.4", "-3.0", "C1=CC=C(C1)C1=Cc2oc3c(c2[SiH2]1)c1c[nH]cc1c1ccc2=CCC=c2c31"
"65553"  , "[SiH2]1C=CC2=C1C=C([SiH2]2)C1=Cc2[se]ccc2[SiH2]1"        , "C12H12SeSi3", "31.4", "6.1", "0.63", "14.8", "-5.2", "1.6", "-3.5", "C1=CC2=C([SiH2]1)C=C([SiH2]2)C1=Cc2[se]ccc2[SiH2]1"
"720918" , "C1C=c2c3ccsc3c3[se]c4cc(oc4c3c2=C1)C1=CC=CC1"            , "C20H12OSSe" , "37.3", "1.9", "0.24", "12.5", "-4.8", "1.8", "-3.0", "C1=CC=C(C1)c1cc2[se]c3c4sccc4c4=CCC=c4c3c2o1"
"1310744", "C1C=CC=C1c1cc2[se]c3c(c4nsnc4c4ccncc34)c2c2ccccc12"      , "C24H13N3SSe", "45.4", "5.6", "0.95", "90.6", "-5.5", "2.0", "-3.5", "C1=CC=C(C1)c1cc2[se]c3c(c4nsnc4c4ccncc34)c2c2ccccc12"
"35232"  , "C1C=CC=C1c1cc2[se]c3c(ncc4ccoc34)c2c2nsnc12"             , "C18H9N3OSSe", "39.3", "5.8", "0.83", "10.1", "-5.5", "1.9", "-3.5", "C1=CC=C(C1)c1cc2[se]c3c(ncc4ccoc34)c2c2nsnc12"
"""


def test_clean_energy_project_1():
    """Tests instantiating dataset."""

    from smlb.datasets.experimental.clean_energy_project.clean_energy_project import (
        CleanEnergyProjectDataset,
    )

    # all data
    with io.StringIO(initial_value=example_input) as source:
        ds = CleanEnergyProjectDataset(source)
        assert ds.num_samples == 7
        assert ds.samples()[6]["id"] == 35232
        assert ds.labels()[1]["HOMO"] == -5.1

    # join
    with io.StringIO(initial_value=example_input) as source:
        ds = CleanEnergyProjectDataset(source, join=True)
        assert ds.num_samples == 6
        assert ds.labels()[0]["HOMO"] == [-5.4, -5.5]

    # samplef and labelf
    with io.StringIO(initial_value=example_input) as source:
        ds = CleanEnergyProjectDataset(
            source, join=True, samplef=lambda e: e["id"], labelf=lambda e: np.sum(e["HOMO"])
        )
        assert ds.samples()[0][0] == [655365, 35232]
        assert ds.labels()[0][0] == -10.9
