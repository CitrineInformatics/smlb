"""Clean Energy Project dataset.

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp 2020, Citrine Informatics.

See class CleanEnergyProjectDataset for details.
"""

from typing import Optional, Sequence, Tuple, Union

import pandas as pd

from smlb import TabularDataFromPandas, params


# todo: support automatic download of dataset from FigShare


class CleanEnergyProjectDataset(TabularDataFromPandas):
    r"""2.3 million molecules with computed photovoltaics-related properties.

    This dataset requires access to the underlying data file (564 MB),
    which can be downloaded from FigShare (last accessed 2020-03-26):

    https://figshare.com/articles/moldata_csv/9640427
    Uploaded by Alán Aspuru-Guzik and Steven Lopez, 2019-08,
    DOI 10.6084/m9.figshare.9640427.v1

    The data is based on the (Harvard) Clean Energy Project:

    Johannes Hachmann, Roberto Olivares-Amaya, Sule Atahan-Evrenk, Carlos Amador-Bedolla,
    Roel S. Sánchez-Carrera, Aryeh Gold-Parker, Leslie Vogt, Anna M. Brockway, Alán Aspuru-Guzik:
    The Harvard Clean Energy Project: Large-Scale Computational Screening and Design
    of Organic Photovoltaics on the World Community Grid, Journal of Physical Chemistry
    Letters 2(17): 2241-2251, 2011. DOI 10.1021/jz200866s

    This dataset, in version v1 as above, contains 2,322,850 molecules (SMILES, stoichiometry)
    with photovoltaics-related properties.
    """

    def __init__(
        self, source: str, join: Optional[Union[str, bool]] = None, **kwargs,
    ):
        """Loads dataset.

        All `IndexedFiniteLabeledDataPandasBackend.__init__` keyword arguments can be passed,
        in particular join, filterf, samplef, and labelf. See there for further explanation.

        Parameters:
            source: path to underlying data file (see class docstring); accepts both
                .csv and .csv.zip versions
            join: whether to join entries with the same chemical sum formula; this changes
                labels from single numbers to varying-length sequences of numbers.
                True can be passed to join by stoichiometry.
            filterf: a function that accepts a sample and returns whether to keep it
                (True) or exclude it (False). Default retains all samples
            samplef: function accepting and returning a sample; applied to all samples
                as post-processing
            labelf: function accepting and returning a label; applied to all labels
                as post-processing

        All samples have these keys:
            id: unique identifier (integer)
            SMILES: SMILES encoding
            formula: stoichiometric formula

        All labels have these keys:
            mass: weight of molecule
            PCE: power conversion efficiency
            VOC: open circuit voltage
            JSC: short-circuit current density
            HOMO: highest occupied molecular orbital
            gap: LUMO-HOMO
            LUMO: lowest unoccupied molecular orbital

        The identifiers and SMILES strings are unique.
        Stoichiometries are not (10,474 unique ones).

        Raises:
            InvalidParameterError: on invalid parameter values
        """

        join = params.any_(join, params.string, params.boolean, params.none)

        # parse boolean settings for join
        if join is True:
            join = "formula"
        if join is False:
            join = None

        data, labels = self._load_data(source)

        super().__init__(data=data, labels=labels, join=join, **kwargs)

    def _load_data(self, source: str) -> Tuple[pd.DataFrame, Sequence[str]]:
        """Load dataset from underlying .csv file.

        Loads rows from CSV file and turns them into a Pandas DataFrame.

        Parameters:
            source: filename; can be .csv or .csv.zip

        Returns:
            Pandas DataFrame, each row corresponding to a molecule,
            and names of label columns
        """

        # load the dataset from file
        # loading via NumPy was significantly slower due to the need for using converters

        df = pd.read_csv(source, skipinitialspace=True)

        # drop tmp_smiles_str column
        del df["tmp_smiles_str"]

        # rename existing columns
        df = df.rename(
            columns={
                "id": "id",
                "SMILES_str": "SMILES",
                "stoich_str": "formula",
                "mass": "mass",
                "pce": "PCE",  # power conversion efficiency
                "voc": "VOC",  # open circuit voltage
                "jsc": "JSC",  # short-circuit current density
                "e_homo_alpha": "HOMO",  # highest occupied molecular orbital
                "e_gap_alpha": "gap",  # LUMO-HOMO
                "e_lumo_alpha": "LUMO",  # lowest unoccupied molecular orbital
            }
        )

        return df, ["mass", "PCE", "VOC", "JSC", "HOMO", "gap", "LUMO"]
