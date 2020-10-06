"""Strehlow & Cook (1973) dataset.

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp 2019, Citrine Informatics.

See class BandGapsStrehlowCook1973Dataset for details.
"""


import json
import os.path
import re
import zipfile

import numpy as np

import smlb


class BandGapsStrehlowCook1973Dataset(smlb.TabularData):
    r"""Band gaps from Strehlow & Cook (1973) dataset.

    Based on:
    W. H. Strehlow, E. L. Cook: Compilation of Energy Band Gaps in Elemental and Binary
    Compound Semiconductors and Insulators, Journal of Physical Chemistry Reference Data
    2(1): 163-199, American Institute of Physics, 1973. DOI 10.1063/1.3253115

    Curated version downloaded from the Citrination platform (https://citrination.com),
    dataset identifier #1160, on 2019-07-17.

    This dataset provides 1,459 compounds, 1,447 of them with measured band gap (eV).
    Compound information includes chemical sum formula and, where specified, measurement
    temperature (K), crystallinity of sample (amorphous, single, or poly-crystalline)
    and other properties.
    """

    # constants for crystallinity categories
    CRYSTALLINITY_NONE = 0  # not specified
    CRYSTALLINITY_AMORPHOUS = 1
    CRYSTALLINITY_SINGLE = 2
    CRYSTALLINITY_POLY = 3

    def _detex_formula(self, s: str) -> str:
        """Helper function, removes TeX markup from a formula string.

        Parameters:
            s: chemical formula that can contain TeX markup from the Citrination platform

        Returns:
            the chemical formula string without the markup
        """

        return s.translate({ord(c): None for c in "_${}"})

    def _canonicalize_formula(self, formula: str) -> str:
        """Helper function, brings chemical sum formula into a (more) canonical form.

        Chemical formulae are sorted by proton number. 
        This eases uniqueness tests and analysis.
        For example, 'CeO2' and 'O2Ce' will be represented as the same string.

        Parameter:
            formula: chemical sum formula. Must not contain TeX markup.

        Returns:
            canonicalized chemical sum formula
        """

        # split into element parts
        element_parts = re.sub(
            r"([A-Z])", r" \1", formula
        ).split()  # insert space before capital letters, then split

        # determine element index and amount
        processed = [re.findall("([a-zA-Z]+)([0-9.]*)", ep)[0] for ep in element_parts]
        elements = np.asarray([smlb.element_data(p[0], "Z") for p in processed])
        amounts = np.asarray([p[1] for p in processed])

        # sort
        order = np.argsort(elements)
        elements, amounts = elements[order], amounts[order]

        # remove unit amounts
        amounts = ["" if a == "1" else a for a in amounts]

        canonical = "".join(
            [smlb.element_data(e, "abbreviation") + a for e, a in zip(elements, amounts)]
        )

        return canonical

    def _parse_json_entry(self, e: dict) -> dict:
        """Helper function, parses a single entry from raw JSON data.

        After processing, the entry contains exactly these keys:
            "formula": canonicalized chemical sum formula
            "band_gap": measured band gap in eV
            "temperature": measurement temperature in Kelvin
            "crystallinity": crystallinity (class constants)
            "ignored_properties": number of ignored other properties
            "ignored_conditions": number of ignored other conditions

        Parameters:
            e: raw JSON entry from the underlying exported data file

        Returns:
            an entry with keys as above
        """

        assert e["category"] == "system.chemical"
        assert e["references"] == [{"doi": "10.1063/1.3253115"}]

        formula = self._detex_formula(e["chemicalFormula"])
        formula = self._canonicalize_formula(formula)

        crystallinity = self.CRYSTALLINITY_NONE
        temperature, band_gap = None, None
        ignored_prop, ignored_cond = 0, 0

        ignored_properties = (
            "Morphology",
            "Temperature derivative of band gap",
            "Color",
            "Electroluminescence",
            "Photoluminescence",
            "Thermoluminescence",
            "Cathodoluminescence",
            "Mechanical luminescence",
            "Phase",
            "Lasing",
        )
        ignored_conditions = ("Transition", "Electric field polarization")

        for prop in e["properties"]:

            # crystallinity
            if prop["name"] == "Crystallinity":
                assert crystallinity == self.CRYSTALLINITY_NONE
                assert len(prop["scalars"]) == 1
                crystallinity = prop["scalars"][0]["value"]

                crystallinity = {
                    "Amorphous": self.CRYSTALLINITY_AMORPHOUS,
                    "Single crystalline": self.CRYSTALLINITY_SINGLE,
                    "Polycrystalline": self.CRYSTALLINITY_POLY,
                }[crystallinity]

            # band gap
            elif prop["name"] == "Band gap":
                assert band_gap is None
                assert len(prop["scalars"]) == 1
                band_gap = float(prop["scalars"][0]["value"])
                assert prop["units"] == "eV"

                if "conditions" in prop:
                    for cond in prop["conditions"]:
                        if cond["name"] == "Temperature":
                            assert temperature is None
                            assert len(cond["scalars"]) == 1
                            temperature = float(cond["scalars"][0]["value"])
                            assert cond["units"] == "K"
                        elif cond["name"] in ignored_conditions:
                            ignored_cond += 1
                        else:
                            raise RuntimeError(f"Unknown condition type '{cond['name']}'")

            # ignored property
            elif prop["name"] in ignored_properties:
                ignored_prop += 1

            else:
                raise RuntimeError(f"Unknown property type '{prop['name']}'")

        return {
            "formula": formula,
            "band_gap": band_gap,
            "temperature": temperature,
            "crystallinity": crystallinity,
            "ignored_properties": ignored_prop,
            "ignored_conditions": ignored_cond,
        }

    def _load_data(self):
        """Helper function, loads dataset from underlying .json file."""

        zip_filename = os.path.join(
            os.path.dirname(__file__), "band_gap_jpcrd_transformed-pif-onlyRef.json.zip"
        )
        filename = "band_gap_jpcrd_transformed-pif-onlyRef.json"

        # load the dataset from file

        with zipfile.ZipFile(zip_filename) as zf:
            with zf.open(filename) as fp:
                raw = json.load(fp)

        # parse entries
        data = [self._parse_json_entry(e) for e in raw]

        return data

    def __init__(
        self, filter_="bg", join=False, samplef=lambda arg: arg, labelf=lambda arg: arg, **kwargs
    ):
        """Loads dataset.

        Parameters control preprocessing.

        Parameters:
            filter_: function that accepts a sample and returns whether to keep it (True)
                     or to exclude it (False). 
                     Possible choices:
                     'all': all entries, including those without band gap, are retained
                     'bg': all entries with a measured band gap are retained
                     't300pm10': all entries with band gap measured at 300 +- 10 K are retained
                     't300pm10_mc': all mono-crystalline entries with band gap measured at 
                                    300 +- 10 K are retained
            join: if True, entries with the same chemical sum formula are joined;
                if a positive integer k, only entries with k or more band gap measurements are retained;
                this changes band gap, temperature and crystallinity from single numbers to
                varying-length sequences of numbers (distributions)
            samplef: function accepting and returning a sample; applied to all samples as post-processing
            labelf: function accepting and returning a label; applied to all labels as post-processing

        Raises:
            InvalidParameterError: on invalid parameter values

        Examples:
            sc = BandGapsStrehlowCook1973Dataset(filter_='t300pm10_mc', join=True, labelf=np.median)

        Note that if joined, there is no need for groups anymore.
        """

        # load data
        data = self._load_data()

        # filter data
        filter_ = smlb.params.enumeration(filter_, {"all", "bg", "t300pm10", "t300pm10_mc"})
        filter_ = {
            "all": lambda _: True,
            "bg": lambda e: e["band_gap"] is not None,
            "t300pm10": lambda e: e["band_gap"] is not None
            and e["temperature"] is not None
            and 290 <= e["temperature"] <= 310,
            "t300pm10_mc": lambda e: e["band_gap"] is not None
            and e["temperature"] is not None
            and 290 <= e["temperature"] <= 310
            and e["crystallinity"] == self.CRYSTALLINITY_SINGLE,
        }[filter_]
        data = [e for e in data if filter_(e)]

        # group data to support stratified sampling
        groups = {None: 0}
        for i, f in enumerate([e["formula"] for e in data]):
            groups[f] = groups.get(f, max(groups.values()) + 1)
            data[i]["group"] = groups[f]
        del groups[None]

        # join data if requested
        # the code below has roughly quadratic runtime. This does not matter for a small
        # dataset like this one, but this solution will not be adequate for larger datasets
        if join is not False:
            join = 1 if join is True else join  # convert True to 1
            join = smlb.params.integer(join, above=0)

            joined_data = []
            for f in groups.keys():  # iterate over unique formulae
                entry = {
                    "formula": f,
                    "band_gap": [],
                    "temperature": [],
                    "crystallinity": [],
                    "ignored_properties": [],
                    "ignored_conditions": [],
                }
                for e in data:
                    if e["formula"] == f:
                        for p in [
                            "band_gap",
                            "temperature",
                            "crystallinity",
                            "ignored_properties",
                            "ignored_conditions",
                        ]:
                            entry[p].append(e[p])
                if len(entry["band_gap"]) >= join:
                    joined_data.append(entry)
            data = joined_data

        # split out band_gap as labels
        labels = [labelf(e["band_gap"]) for e in data]
        for i in range(len(data)):
            del data[i]["band_gap"]
            data[i] = samplef(data[i])

        # initialize state
        super().__init__(data=np.array(data), labels=np.array(labels), **kwargs)
