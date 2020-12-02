"""Citrine collection of superconductors (2016) dataset.

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp 2020, Citrine Informatics.

See class SuperconductorsCitrine2016Dataset for details.
"""


import os.path
from typing import Any, Callable

import numpy as np

from smlb import BenchmarkError, TabularData, params


# todo: process non-standard formulas

# todo: canonicalize formulas for joining


class SuperconductorsCitrine2016Dataset(TabularData):
    r"""Superconductors with critical temperature, Citrine literature collection, 2016.

    Based on:
    Dataset id 2210, created 2016-10-29 by Kyle Michel.
    Collected and compiled by Citrine Informatics.
    https://citrination.com/data_views/14083/data_summary

    This dataset contains chemical formulas and superconducting critical temperatures
    collected from the literature. Each of the 587 entries contains
    one or two references (URL), chemical formula (string) and superconducting temperature T_c (K).

    Notes:
    * The dataset description states 586 entries. Reference 1 states 546 entries.

    References:
    [1] Bryce Meredig, Erin Antono, Carena Church, Maxwell Hutchinson, Julia Ling,
        Sean Paradiso, Ben Blaiszik, Ian Foster, Brenna Gibbons, Jason Hattrick-Simpers,
        Apurva Mehta, Logan Ward: Can machine learning identify the next high-temperature
        superconductor? Examining extrapolation performance for materials discovery,
        Molecular Systems Design & Engineering 3(5): 819-825, 2018. DOI 10.1039/c8me00012c
    """

    def __init__(
        self,
        process: bool = True,
        join: bool = True,
        filter_: Callable[[dict], bool] = lambda _: True,
        samplef: Callable[[dict], dict] = lambda arg: arg,
        labelf: Callable[[float], Any] = lambda arg: arg,
        **kwargs,
    ):
        """Loads dataset.

        Parameters control preprocessing. Order:
        processing, joining, filtering, sample and label transform.

        Parameters:
            process: if False, entries are passed as-are; in particular, some formulas
                will contain variables (AxB1-x) and brackets; some labels will be
                intervals (from,to); if True, formulas are turned into simple
                sum formulas (no variables, no brackets) and all labels will be numbers
                CURRENTLY, such formulas are only flagged, but not parsed; only labels change
            join: whether to join entries with the same chemical sum formula; this changes
                labels from single numbers to varying-length sequences of numbers
            filter_: a function that accepts a sample and returns whether to keep it
                (True) or exclude it (False). Default retains all samples
            samplef: function accepting and returning a sample; applied to all samples
                as post-processing
            labelf: function accepting and returning a label; applied to all labels
                as post-processing

        A conservative parametrization is:
        SuperconductorsCitrine2016Dataset(
            process=True, join=True,
            filter_=lambda e: not any(e["flagged_formula"]),
            samplef=lambda e: e["formula"],
            labelf=lambda tc: np.median(tc)
        )
        This results in a dataset of valid formulas with Tc as labels.

        All entries have these keys:
            "citation1": first citation URL
            "citation2": second citation URL if it exists, empty string otherwise
            "formula": chemical sum formula
            "Tc/K": superconducting critical temperature in K
            "process_Tc/K": True if label was changed in processing, False otherwise
            "process_formula": True if formula was changed in processing, False otherwise
            "flagged_formula": True if formula was flagged for some reason,
                including presence of variables (x, y) or unclear notation "+d"
        These entries can be used to filter.

        Raises:
            InvalidParameterError: on invalid parameter values

        Examples:
            sc = SuperconductorsCitrine2016Dataset()
            sc = SuperconductorsCitrine2016Dataset(process=True, filter_=lambda e: not e['flagged_formula'])
        """

        # todo: params test for functions with signature
        process = params.boolean(process)
        join = params.boolean(join)

        # load data
        data = self._load_data()

        # process data if requested
        if process:
            data = [self._process(e) for e in data]

        # join data if requested
        # the code below has roughly quadratic runtime. This does not matter for a small
        # dataset like this one, but this solution will not be adequate for larger datasets
        if join:
            # group data by unique formula
            # todo: canonicalize formula
            groups = {None: 0}
            for i, f in enumerate([e["formula"] for e in data]):
                groups[f] = groups.get(f, max(groups.values()) + 1)
                data[i]["group"] = groups[f]
            del groups[None]

            joined_data = []
            for f in groups.keys():  # iterate over unique formulae
                entry = {
                    "formula": f,
                    "citation1": [],
                    "citation2": [],
                    "Tc/K": [],
                    "process_Tc/K": [],
                    "process_formula": [],
                    "flagged_formula": [],
                }
                for e in data:
                    if e["formula"] == f:
                        for p in [
                            "citation1",
                            "citation2",
                            "Tc/K",
                            "process_Tc/K",
                            "process_formula",
                            "flagged_formula",
                        ]:
                            entry[p].append(e[p])
                joined_data.append(entry)
            data = joined_data

        # filter data
        data = [e for e in data if filter_(e)]

        # split out T_c as labels
        labels = [labelf(e["Tc/K"]) for e in data]
        for i in range(len(data)):
            del data[i]["Tc/K"]
            data[i] = samplef(data[i])

        # initialize state
        super().__init__(data=np.array(data), labels=np.array(labels), **kwargs)

    def _process(self, e: dict) -> dict:
        """Helper function, process single entry.

        Transforms entry as follows:
        * replaces label ranges (a,b) with their mean
        * marks entries containing a variable

        Adds these keys:
            "process_Tc/K": True if label was changed in processing, False otherwise
            "process_formula": True if formula was changed in processing, False otherwise
            "flagged_formula": True if formula was flagged for some reason,
                including presence of variables (x, y) or unclear notation "+d"
        """

        e["process_Tc/K"] = False
        e["process_formula"] = False
        e["flagged_formula"] = False

        # replace label intervals by mean
        if not isinstance(e["Tc/K"], float):
            e["Tc/K"] = (e["Tc/K"][1] - e["Tc/K"][0]) / 2
            e["process_Tc/K"] = True

        # currently no processing of formulas

        # mark non-standard formulas
        flags = ("x", "y", "+d", "-d", ":C", "2H-", "Hg-", ",", "TL 2201", "LSCO", "*", "-")
        if any(s in e["formula"] for s in flags):
            e["flagged_formula"] = True

        return e

    def _parse_csv_entry(self, e: str) -> dict:
        """Helper function, parse single entry from underlying data.

        After processing, the entry contains exactly these keys:
            "citation1": first citation URL
            "citation2": second citation URL if it exists, empty string otherwise
            "formula": chemical sum formula
            "Tc/K": superconducting critical temperature in K

        Parameters:
            e: line from the underlying exported data file

        Returns:
            an entry with keys as above
        """

        e = e.split(",")

        if len(e) != 6:
            # some entries have commas in the formula
            formula = ",".join(e[2:-3])
            if formula[0] == '"':
                formula = formula[1:-1]
            e = e[:2] + [formula,] + e[-3:]

        assert len(e) == 6
        assert e[3] == "Superconducting critical temperature (Tc)"
        assert e[5] == "K"

        try:
            value = float(e[4])
        except ValueError:
            try:
                value = e[4].split(" to ")
                assert len(value) == 2
                value = (float(value[0]), float(value[1]))
            except Exception as e:
                raise BenchmarkError(f"invalid temperature '{value}'") from e

        return {
            "citation1": e[0],
            "citation2": e[1],
            "formula": e[2],
            "Tc/K": value,
        }

    def _load_data(self):
        """Helper function, loads dataset from underlying .csv file.

        Loads rows from CSV file and turns them into dictionaries.
        """

        filename = os.path.join(os.path.dirname(__file__), "superconductors_transformed.csv")

        # load the dataset from file

        with open(filename) as fp:
            raw = fp.read()

        # parse entries
        raw = raw.splitlines()[1:]  # drop header line with columns
        assert len(raw) == 588, f"datafile '{filename}' changed unexpectedly"
        data = [self._parse_csv_entry(e) for e in raw]

        return data
