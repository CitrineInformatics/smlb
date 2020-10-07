"""Ni-Superalloy dataset.

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
2019, Brendan Folie, Citrine Informatics.

See class NiSuperalloyDataset for details.
"""


import os
import json
import zipfile
from typing import List, Optional, Tuple, Union
import numpy as np

from smlb import InvalidParameterError, params, TabularData


class NiSuperalloyDataset(TabularData):
    """
    Ni-Superalloy dataset.

    Based on:
    Bryce D. Conduit, Nicholas G. Jones, Howard J. Stone, Gareth John Conduit:
    Design of a nickel-base superalloy using a neural network, Materials & Design 131: 358-365,
    Elsevier, 2017. DOI 10.1016/j.matdes.2017.06.007

    The dataset was downloaded from the Citrination platform (https://citrination.com),
    dataset identifier #153493, Version 10.

    There are 2800 rows.
    The data have columns for composition (25 elements are present in at least one row),
    whether the alloy was powder processed (0 or 1), whether it was pressure treated (0 or 1),
    heat treatment time (hours) and temperature (degrees Celcius) for up to 4 heat treatment steps,
    the total time spent in heat treatment (hours), the maximum heat treatment temperature
    (degrees Celcius), and the area under the time-temperature curve (degrees Celcius * hours).
    A value of 0 generally implies that the heat treatment step was not done, but there
    are some missing values. The total time and max temperature are generally more reliable
    than the individual heating steps. The total compositions do not always add up to 100%,
    but with about a dozen exceptions they always add up to somewhere between 95% and 105%.

    There are also three columns for a pressure treatment step (temperature, time, pressure),
    but since only 51 rows have non-zero entries, this information is not used.

    There are 5 labels: ultimate tensile strength (MPa), elongation (unitless), stress rupture
    stress (MPa), stress rupture time (hours), and yield strength (MPa). Tensile strength and
    elongation occur together in 898 rows, stress rupture stress and time occur together in
    856 rows, and yield strength occurs in 1046 rows. 898+856+1046=2800, so every row has exactly
    one output set. The other values are denoted as NaN.

    """

    DEFAULT_PATH = os.path.split(os.path.realpath(__file__))[0] + "/ni_superalloys_3.json.zip"

    POWDER_PROCESSED_NO = 0
    POWDER_PROCESSED_YES = 1

    def __init__(
        self, labels_to_load: Optional[Union[str, List[str]]] = None, ignore_dubious: bool = False
    ):
        """Initialize Ni-superalloy dataset with specified labels.

        Parameters:
            labels_to_load (str or List[str]): which labels to load. Options are
                'Yield Strength', 'Ultimate Tensile Strength', 'Stress Rupture Time',
                'Stress Rupture Stress', and 'Elongation'.
                If None, then all labels are loaded.
            ignore_dubious: whether or not to ignore samples that have something
                questionable about them

        """

        labels_to_load = params.optional_(
            labels_to_load,
            lambda arg: params.any_(
                arg,
                params.string,
                lambda arg: params.sequence(arg, type_=str),
            ),
        )
        ignore_dubious = params.boolean(ignore_dubious)

        filepath = self.DEFAULT_PATH
        data, labels = self._load_data_and_labels(filepath, labels_to_load, ignore_dubious)
        super().__init__(data=data, labels=labels)

    def _load_data_and_labels(
        self,
        filepath: str,
        labels_to_load: Optional[List[str]] = None,
        ignore_dubious: bool = False,
    ):
        """Load data and labels from .json file."""
        raw = self._unzip_json_file(filepath)
        if ignore_dubious:
            raw = [e for e in raw if self._filter_dubious(e)]
        # dtype=object is necessary because this is a mixed-type array (float and string)
        data = np.array([self._parse_json_data(e) for e in raw], dtype=object)
        labels = np.array([self._parse_json_labels(e, labels_to_load) for e in raw], dtype=float)
        return data, labels

    @staticmethod
    def _unzip_json_file(filepath: str):
        """Open and read zipped json file."""
        filename = os.path.basename(filepath)
        assert (
            filename[-4:] == ".zip"
        ), f"File path must point to a .zip file, instead got '{filepath}'"

        with zipfile.ZipFile(filepath) as zf:
            unzipped_filename = filename[:-4]
            with zf.open(unzipped_filename) as fp:
                raw = json.load(fp)
        return raw

    @staticmethod
    def _extract_raw_composition(entry: dict) -> List[dict]:
        """Get composition in its raw form."""
        raw_composition = entry.get("composition")
        if raw_composition is None or not isinstance(raw_composition, list):
            raise InvalidParameterError(
                expected="Chemical composition as a list", got=raw_composition
            )
        return raw_composition

    @staticmethod
    def _filter_dubious(entry: dict) -> bool:
        """
        Determine whether or not a json entry has something questionable about it.
        Currently, the only thing filtered on is if the composition has an asterisk in it,
        which occurs for 6 samples.

        Parameters:
            entry (dict): A json entry corresponding to a row in the dataset.

        Returns: bool
            True if the composition contains an asterisk.

        """
        raw_composition = NiSuperalloyDataset._extract_raw_composition(entry)
        composition_dict = NiSuperalloyDataset._parse_composition_as_dict(raw_composition)
        composition_dict_float, exception_caught = NiSuperalloyDataset._dict_values_to_float(
            composition_dict
        )
        return not exception_caught

    def _parse_json_data(self, entry: dict):
        """
        Helper function to parse data in a single row from the raw json.

        Parameters:
            entry (dict): A json entry corresponding to a row in the dataset.

        Returns: array
            Array of data in this row.

        """
        assert entry["category"] == "system.chemical"
        raw_composition = NiSuperalloyDataset._extract_raw_composition(entry)
        composition: str = self._parse_composition(raw_composition)

        properties = entry.get("properties")
        if properties is None or not isinstance(properties, list):
            raise InvalidParameterError(
                expected="A list of dictionaries, one for each property", got=properties
            )
        heat_treatment_1_time = self._get_scalar_property(
            properties, "Heat treatment 1 Time", units="hours", default_value=0
        )
        heat_treatment_1_temp = self._get_scalar_property(
            properties, "Heat treatment 1 Temperature", units="$^{\\circ}$C", default_value=0
        )
        heat_treatment_2_time = self._get_scalar_property(
            properties, "Heat treatment 2 Time", units="hours", default_value=0
        )
        heat_treatment_2_temp = self._get_scalar_property(
            properties, "Heat treatment 2 Temperature", units="$^{\\circ}$C", default_value=0
        )
        heat_treatment_3_time = self._get_scalar_property(
            properties, "Heat treatment 3 Time", units="hours", default_value=0
        )
        heat_treatment_3_temp = self._get_scalar_property(
            properties, "Heat treatment 3 Temperature", units="$^{\\circ}$C", default_value=0
        )
        heat_treatment_4_time = self._get_scalar_property(
            properties, "Heat treatment 4 Time", units="hours", default_value=0
        )
        heat_treatment_4_temp = self._get_scalar_property(
            properties, "Heat treatment 4 Temperature", units="$^{\\circ}$C", default_value=0
        )
        total_heat_treatment_time = self._get_scalar_property(
            properties, "Total heat treatment time", units="hours"
        )
        max_heat_treatment_temp = self._get_scalar_property(
            properties, "Max Heat Treatment Temperature", units="$^{\\circ}$C"
        )
        area_under_heat_treatment_curve = self._get_scalar_property(
            properties, "Area under heat treatment curve", units="$^{\\circ}$C * hours"
        )

        powder_processed_dict = {"No": self.POWDER_PROCESSED_NO, "Yes": self.POWDER_PROCESSED_YES}
        powder_processed = self._get_categorical_property(
            properties, "Powder processed", categories_dict=powder_processed_dict
        )

        data_array = [
            composition,
            heat_treatment_1_time,
            heat_treatment_1_temp,
            heat_treatment_2_time,
            heat_treatment_2_temp,
            heat_treatment_3_time,
            heat_treatment_3_temp,
            heat_treatment_4_time,
            heat_treatment_4_temp,
            total_heat_treatment_time,
            max_heat_treatment_temp,
            area_under_heat_treatment_curve,
            powder_processed,
        ]
        return data_array

    def _parse_json_labels(self, entry: dict, labels_to_load: Optional[List[str]] = None):
        """
        Helper function to parse labels in a single row from the raw json.

        Parameters:
            entry (dict): A json entry corresponding to a row in the dataset.
            labels_to_load (List[str]): Optional list of labels to load.

        Returns: array
            Array of labels in this row that we are interested in.

        """
        if labels_to_load is None:
            labels_to_load = [
                "Yield Strength",
                "Ultimate Tensile Strength",
                "Stress Rupture Time",
                "Stress Rupture Stress",
                "Elongation",
            ]

        properties = entry.get("properties")
        if properties is None or not isinstance(properties, list):
            raise InvalidParameterError(
                expected="A list of dictionaries, one for each property", got=properties
            )

        labels_array = []
        for label in labels_to_load:
            labels_array.append(self._get_scalar_property(properties, label, default_value=None))
        return labels_array

    @staticmethod
    def _parse_composition(raw_composition: List[dict]) -> str:
        """
        Helper function to parse composition as a string.

        Parameters:
            raw_composition (List[dict]): A list, each entry of which corresponds to an element.
                An entry is a dict with an 'element' key and an 'idealWeightPercent' key.
                The element is a string (e.g., 'Cu') and the weight percent is another dict
                with a single key, 'value', pointing to a floating point number.
                The values are in percentage points, and add up to ~100.

        Returns: str
            Chemical composition as string, e.g. 'Al5.5Ni94.0W0.5'

        """
        composition_dict = NiSuperalloyDataset._parse_composition_as_dict(raw_composition)
        composition_dict_float, _ = NiSuperalloyDataset._dict_values_to_float(composition_dict)
        composition_str: str = ""
        for element_name, element_amount in composition_dict_float.items():
            if element_amount > 0:
                composition_str += element_name + str(element_amount)
        return composition_str

    @staticmethod
    def _parse_composition_as_dict(raw_composition: List[dict]) -> dict:
        """
        Helper function to parse composition as a dictionary.

        Parameters:
            raw_composition (List[dict]): A list, each entry of which corresponds to an element.
                An entry is a dict with an 'element' key and an 'idealWeightPercent' key.
                The element is a string (e.g., 'Cu') and the weight percent is another dict
                with a single key, 'value', pointing to a floating point number.
                The values are in percentage points, and add up to ~100 (but not exactly).

        Returns: dict
            Chemical composition as a dictionary with the elements as keys
                and their raw amounts as values

        """
        composition_dict = dict()
        for entry in raw_composition:
            try:
                element_name = entry["element"]
                element_amount = entry["idealWeightPercent"]["value"]
            except KeyError:
                raise InvalidParameterError(
                    expected="Element amount as a dictionary of the form\n"
                    "{'element': <element name>,"
                    "'idealWeightPercent': "
                    "{'value': <element amount>}}",
                    got=entry,
                )
            composition_dict[element_name] = element_amount
        return composition_dict

    @staticmethod
    def _dict_values_to_float(d: dict) -> Tuple[dict, bool]:
        """
        Convert a dictionary's values to their floating point representations, if possible.

        Parameters:
            d: a dictionary

        Returns: dict, bool
            A modified version of `d`, and a boolean flag indicating whether or not
            an Exception was caught
        """
        d_copy = dict()
        exception_caught = False
        for key, value in d.items():
            try:
                value_float = float(value)
            except ValueError:
                exception_caught = True
                value_float = NiSuperalloyDataset._parse_peculiar_amount(value)
            d_copy[key] = value_float
        return d_copy, exception_caught

    @staticmethod
    def _parse_peculiar_amount(x: str) -> float:
        """
        Deals with dataset-specific-peculiarities in composition amounts.

        Some composition amounts have a trailing asterisk, e.g., '2*'. The meaning is unclear.
            Perhaps it denotes that the amount is imprecise. In any case, they only occur in 6
            samples. The trailing asterisk will be ignored.

        """
        if x[-1] == "*":
            x = x[:-1]
        try:
            return float(x)
        except ValueError:
            raise InvalidParameterError("Amount as a float", x)

    def _get_scalar_property(
        self,
        properties: List[dict],
        property_name: str,
        units: Optional[str] = None,
        default_value: Optional[float] = None,
    ) -> float:
        """
        A helper function to get a single scalar property.
        This calls _get_single_property and then checks that the result can be
        turned into a float.

        Parameters:
            properties: A list of dicts, each of which is a single property.
            property_name: The name of the property to get the value of.
            units: Optional expected units string.
            default_value: Value to return if `property_name` is not present.

        Raises:
            InvalidParameterError: if the value cannot be expressed as a float

        Returns: float
            The value of the desired property.

        """
        try:
            val = self._get_single_property(properties, property_name, units, default_value)
            if val is None:
                return None
            return float(val)
        except ValueError:
            raise InvalidParameterError(
                expected=f"Property {property_name} should have a value "
                f"that can be expressed as a float",
                got=properties,
            )

    def _get_categorical_property(
        self, properties: List[dict], property_name: str, categories_dict: dict
    ) -> int:
        """
        Helper function to get a single categorical property as an int.

        Parameters:
            properties: A list of dicts, each of which is a single property.
            property_name: The name of the property to get the value of.
            categories_dict: Dict from the categorical property (string) to a unique integer value.

        Raises:
            InvalidParameterError: if the value is not in the expected list of possible categories
                as given by the keys in `categories_dict`

        Returns: int
            An integer that corresponds to the value of the desired property.

        """
        category = self._get_single_property(properties, property_name)
        try:
            return categories_dict[category]
        except KeyError:
            raise InvalidParameterError(
                f"A value in the array: {categories_dict.keys()}", category
            )

    @staticmethod
    def _get_single_property(
        properties: List[dict], property_name: str, units: Optional[str] = None, default_value=None
    ):
        """
        Helper function to get a single property.

        Parameters:
            properties: A list of dicts, each of which is a single property. Each entry is expected
             to have a 'name' field that corresponds to the property name and a `scalars` field
                that is a list with one entry, a dict of the form {'value': <property value>}.
                It may also have a 'units' field.
            property_name: The name of the property to get the value of. `properties` is expected
                to have exactly one entry with the 'name' field equal to `property_name`.
            units: Optional expected value of 'units' field. If specified, then there must be a
                'units' field and its value must correspond to `units`.
            default_value: Value to return if `property_name` is not present.

        Raises:
            InvalidParameterError: if `properties` does not conform to the expected structure

        Returns:
            The value of the property `property_name`

        """
        matching_props = [prop for prop in properties if prop.get("name") == property_name]
        if len(matching_props) == 0:
            return default_value
        elif len(matching_props) > 1:
            raise InvalidParameterError(
                expected=f"Only one entry in properties should have name" f" '{property_name}'",
                got=properties,
            )
        matching_prop = matching_props[0]

        try:
            scalars = matching_prop["scalars"]
            assert len(scalars) == 1
            val = scalars[0]["value"]
            if units is not None:
                assert matching_prop["units"] == units
        except (KeyError, AssertionError):
            units_str = "" if units is None else f", 'units': {units}"
            raise InvalidParameterError(
                expected="Property as a dictionary of the form\n"
                "{'name': <property name>, 'scalars': "
                "[{'value': <property value>}]" + units_str + "}",
                got=matching_prop,
            )
        return val
