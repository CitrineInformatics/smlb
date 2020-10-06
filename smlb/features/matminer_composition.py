"""Matminer composition-based materials features.

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
2019-2020, Citrine Informatics.

See docstring of MatminerCompositionFeatures for details.

Depends on packages `pymatgen` and `matminer`.
"""

from typing import Any, Callable, Sequence, Union

import numpy as np

from smlb import (
    BenchmarkError,
    Data,
    Features,
    params,
    TabularData,
)

# todo: ionic features with fast=False (matminer's default) setting break for non-integer
#       compositions. this can either be accepted, or formulas can be lifted by
#       multiplying amounts by appropriate powers of 10 before passing them to matminer.
#       example: Ni 1.8 W .05 Al 0.4 -> Ni 180 W 5 Al 40  (spaces for readability)


class MatminerCompositionFeatures(Features):
    """Matminer composition-based materials features.

    Based on the matminer package.

    Reference:
        Logan Ward, Alexander Dunn, Alireza Faghaninia, Nils E.R. Zimmermann, Saurabh Bajaj,
        Qi Wang., Joseph Montoya, Jiming Chen, Kyle Bystrom, Maxwell Dylla, Kyle Chard,
        Mark Asta, Kristin A. Persson, G. Jeffrey Snyder, Ian Foster, Anubhav Jain:
        Matminer: An open source toolkit for materials data mining, Computational Materials
        Science 152: 60--69, Elsevier, 2018. DOI 10.1016/j.commatsci.2018.05.018

    Code and documentation:
        https://hackingmaterials.lbl.gov/matminer/
        https://github.com/hackingmaterials/matminer

    Currently supports four types of features:
    * Stoichiometric attributes describe the amount of each element present in
      a compound using several L^p norms.
    * Elemental property statistics, computed on 22 different elemental
      properties, such as distribution of atomic radii.
    * Ionic compound attributes, such as whether it is possible to form an
      ionic compound from the elements, and the fractional "ionic character" of the compound.
    * Electronic structure attributes, which are the average fraction of
      electrons from the s, p, d and f valence shells.
    """

    def __init__(
        self,
        select: Union[str, Sequence[str]] = "all",
        samplef: Callable[[Any], Any] = lambda arg: arg,
        stoichiometry_p_list: Sequence[int] = (0, 2, 3, 5, 7, 10),
        elemental_preset: str = "magpie",
        ionic_fast: bool = False,
        valence_orbitals: Sequence[str] = ("s", "p", "d", "f"),
        valence_props: Sequence[str] = ("avg", "frac"),
        **kwargs,
    ):
        """Initialize state.

        Selected parameters of the wrapped matminer classes Stoichiometry, ElementProperty,
        IonProperty, ValenceOrbital can be passed through. These parameters are prefixed
        with stoichiometry, elemental, ionic, valence. For example, stoichiometry_p_list
        is the p_list parameter of Stoichiometry. For further details on these, see
        https://github.com/hackingmaterials/matminer/blob/master/matminer/featurizers/composition.py

        Parameters:
            select: which feature sets to compute (by default, all). Specifying 
                multiple sets (e.g., ('stoichiometry', 'elemental') selects both).
                Valid choices:
                'all': all features
                'stoichiometry': norms of stoichiometric features
                'elemental': element properties
                'ionic': ion properties
                'valence': valence orbital shell features
            samplef: a function accepting and returning a sample. This enables
                transformation of samples, for example, to select an entry by key 
                if sample is a dictionary, or to turn a dictionary into a vector.
                Default is to return the sample unchanged.
            stoichiometry_p_list: list of L_p norms to compute
            elemental_preset: matminer preset to use. Valid choices include:
                'magpie', 'deml', 'matminer', 'matscholar_el', 'megnet_el'
            ionic_fast: if True, assumes that elements exist in single oxidation state
            valence_orbitals: which valence orbitals to consider
            valence_props: whether to return average properties, fractional, or both

        Requires the matminer package (see file documentation).
        """

        super().__init__(**kwargs)

        SELECT_SETS = ("stoichiometry", "elemental", "ionic", "valence")

        if select == "all":
            select = SELECT_SETS
        if isinstance(select, str):
            select = (select,)  # tuple(str,) yields tuple of characters in str
        select = params.tuple_(select, lambda arg: params.enumeration(arg, set(SELECT_SETS)),)

        self._stoichiometry_p_list = params.tuple_(
            stoichiometry_p_list, lambda p: params.integer(p, from_=0)
        )
        self._elemental_preset = params.enumeration(
            elemental_preset, {"magpie", "deml", "matminer", "matscholar_el", "megnet_el"}
        )
        self._ionic_fast = params.boolean(ionic_fast)
        self._valence_orbitals = params.tuple_(
            valence_orbitals, lambda arg: params.enumeration(arg, {"s", "p", "d", "f"})
        )
        self._valence_props = params.tuple_(
            valence_props, lambda arg: params.enumeration(arg, {"avg", "frac"})
        )

        self.samplef = samplef  # todo: add callable to params

        # set up matminer
        try:
            import matminer
            import matminer.featurizers
            import matminer.featurizers.base
            import matminer.featurizers.composition
            import matminer.featurizers.conversions
            import pymatgen
        except ModuleNotFoundError as e:
            raise BenchmarkError(
                f"'{type(self).__name__}' requires 'matminer' and 'pymatgen' packages"
            ) from e

        self._composition = pymatgen.Composition

        # set up features
        features = []
        if "stoichiometry" in select:
            features.append(
                matminer.featurizers.composition.Stoichiometry(p_list=self._stoichiometry_p_list)
            )
        if "elemental" in select:
            features.append(
                matminer.featurizers.composition.ElementProperty.from_preset(
                    self._elemental_preset
                )
            )
        if "ionic" in select:
            features.append(matminer.featurizers.composition.IonProperty(fast=self._ionic_fast))
        if "valence" in select:
            features.append(
                matminer.featurizers.composition.ValenceOrbital(
                    orbitals=self._valence_orbitals, props=self._valence_props
                )
            )

        self._mmfeatures = matminer.featurizers.base.MultipleFeaturizer(features)

    def apply(self, data: Data) -> TabularData:
        """Compute matminer composition-based materials features.

        Parameters:
            data: material compositions, given as sum formula strings
                  Can be labeled, and labels will be retained

        Returns:
            TabularData or TabularLabeledData with matminer composition-based
            materials features as samples
        """

        data = params.instance(data, Data)

        inputs_ = tuple(self._composition(self.samplef(s)) for s in data.samples())
        features = self._mmfeatures.featurize_many(inputs_, pbar=False)
        features = np.asfarray(features)

        result = TabularData(data=features, labels=data.labels() if data.is_labeled else None)

        return result
