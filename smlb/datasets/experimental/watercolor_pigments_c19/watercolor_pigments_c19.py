"""Watercolor pigments dataset.

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp 2019-2020, Citrine Informatics.

Provides the data itself and specific features for them.
See classes `WatercolorPigments2019Dataset` and `WatercolorPigments2019DataFeatures` for details.
"""

import os.path
import zipfile

import numpy as np

from smlb import InvalidParameterError, params, TabularData, Features


class WatercolorPigments2019Dataset(TabularData):
    r"""National Taiwan University Watercolor Pigments Spectral Measurements (NTU WPSM, Chen 2019) dataset.

    Measured colors of binary mixtures of watercolor pigments on white paper.

    Reflectance of 780 binary mixtures of 13 primary watercolor pigments described by reflectance and transmittance.
    Measurements use an OTO SD1220 spectrometer, a light source box and K1 light source.
    Contains two types of data, increasing amounts of one primary pigment (I) and 1:1, 1:2, 2:1 mixtures of two
    pigments (M). Note that absolute pigment amounts matter, that is, 0.01 mL of pigment A and B each is different
    from 0.02 mL of A and B each. This is due to transparency of watercolor pigments. 12 increasing quantitites of
    each primary pigment (0.01 mL, ..., 0.10 mL, 0.12 mL, 0.16 mL) and 780 out of 3 (1:1, 1:2 ratios) x 3 (amounts)
    x 13 x 13 = 1404 possible binary mixtures. The dataset actually only provides colors for the primary pigments;
    in particular, transmittance and reflectance of pigment and reflectance of substrate are not provided separately.

    Based on:

    Mei-Yun Chen: Prediction Model for Semitransparent Watercolor Pigment Mixtures Using Deep Learning
    with a Dataset of Transmittance and Reflectance, PhD thesis, Graduate Institute of Networking and
    Multimedia, College of Electrical Engineering and Computer Science, National Taiwan University,
    Taipei, Taiwan, January 2019.

    Mei-Yun Chen, Ya-Bo Huang, Sheng-Ping Chang, Ming Ouhyoung: Prediction Model for Semitransparent
    Watercolor Pigment Mixtures Using Deep Learning with a Dataset of Transmittance and Reflectance,
    arXiv: 1904.00275, 2019. https://arxiv.org/abs/1904.00275

    Based on file "NTU_WPSM_dataset.rar", downloaded 2020-03-05 from http://www.cmlab.csie.ntu.edu.tw/~meiyun/
    """

    # color names (manufacturer: Winsor & Newton)
    COLOR_NAMES = (
        None,  # enable use of 1-based indices as in dataset
        "cadmium red",  # P1
        "alizarin crimson",  # P2
        "burnt sienna",  # P3
        "lemon yellow",  # P4
        "cadmium yellow",  # P5
        "raw sienna",  # P6
        "sap green",  # P7
        "cerulean blue",  # P8
        "cobalt blue",  # P9
        "ultramarine",  # P10
        "prussian blue",  # P11
        "ivory black",  # P12
        "chinese white",  # P13
    )

    def _load_data(self):
        """Load raw data from underlying .zip file."""

        # load raw data from archive

        zip_filename = os.path.join(os.path.dirname(__file__), "NTU_WPSM_dataset.zip")

        def loadcsv(zf, filename: str):
            with zf.open(filename, mode="r") as fp:
                raw = fp.read().decode("ascii")
                return tuple(line.split(",") for line in raw.split("\r\n"))

        with zipfile.ZipFile(zip_filename) as zf:
            raw_pr = loadcsv(zf, "PigmentRGB_primary.csv")
            raw_11 = loadcsv(zf, "PigmentRGB_mixed_1_1.csv")
            raw_12 = loadcsv(zf, "PigmentRGB_mixed_1_2.csv")
            raw_21 = loadcsv(zf, "PigmentRGB_mixed_2_1.csv")

        # parse entries

        assert (
            raw_pr[0] == ["", "Pigment index", "R", "G", "B"]
            and raw_pr[-1] == [""]
            and len(raw_pr) == 158
        )
        assert (
            raw_11[0] == ["", "PigmentA", "PigmentB", "QuantityA", "QuantityB", "R", "G", "B"]
            and raw_11[-1] == [""]
            and len(raw_11) == 258
        )
        assert raw_12[0] == ["", "R", "G", "B"] and raw_12[-1] == [""] and len(raw_12) == 200
        assert (
            raw_21[0] == ["", "PigmentA", "PigmentB", "QuantityA", "QuantityB", "R", "G", "B"]
            and raw_21[-1] == [""]
            and len(raw_21) == 192
        )

        def concf(s):
            """Extract concentration from identifier"""
            assert s[-7] == "_" and s[-2:] == "ml"
            return float(s[-6:-2])

        def indf(s):
            """Extract color index from identifier 'Pi'"""
            assert s[0] == "P"
            return int(s[1:])

        # primary pigments P1, ..., P13
        # these are increasing concentrations of the same color
        primary = tuple(
            {
                "type": "primary",
                "identifier": line[0],
                "concentration": concf(line[0]),  # in mL
                "index": indf(line[1]),
                "rgb": (int(line[2]), int(line[3]), int(line[4])),
            }
            for line in raw_pr[1:-1]
        )

        mixed_11 = tuple(
            {
                "type": "mixture11",
                "identifier": line[0],
                "indexA": indf(line[1]),
                "indexB": indf(line[2]),
                "concentrationA": float(line[3]),
                "concentrationB": float(line[4]),
                "rgb": (int(line[5]), int(line[6]), int(line[7])),
            }
            for line in raw_11[1:-1]
        )
        mixed_21 = tuple(
            {
                "type": "mixture21",
                "identifier": line[0],
                "indexA": indf(line[1]),
                "indexB": indf(line[2]),
                "concentrationA": float(line[3]),
                "concentrationB": float(line[4]),
                "rgb": (int(line[5]), int(line[6]), int(line[7])),
            }
            for line in raw_21[1:-1]
        )

        def transff(line, other):
            """Extract entry from incomplete row."""
            id_ = line[0]
            idprefix = id_[:-3]  # drop fixed-length post-fix _x_y
            index = tuple(e["identifier"][:-3] for e in other).index(idprefix)
            return {
                "type": "mixture12",
                "identifier": id_,
                "indexA": other[index]["indexA"],
                "indexB": other[index]["indexB"],
                "concentrationA": float(id_[-3]) / 100,
                "concentrationB": float(id_[-1]) / 100,
                "rgb": (int(line[1]), int(line[2]), int(line[3])),
            }

        mixed_12 = tuple(transff(line, mixed_21) for line in raw_12[1:-1])

        return primary + mixed_11 + mixed_12 + mixed_21

    def __init__(
        self, filter_="mixture", samplef=lambda arg: arg, labelf=lambda arg: arg, **kwargs
    ):
        """Loads dataset.

        Parameters control preprocessing.

        Parameters:
            filter_: a function that accepts a sample and returns whether to keep it (True) or exclude it (False):
                     Pre-defined choices:
                     'all': all entries (all primary pigment concentrations and all mixtures)
                     'primary': all primary pigment concentrations
                     'mixture': all mixtures of two different pigments
            samplef: function accepting and returning a sample; applied to all samples as post-processing
            labelf: function accepting and returning a label; applied to all labels as post-processing

        Raises:
            InvalidParameterError: on invalid parameter values

        Examples:
            wc = WatercolorPigments2019Dataset(variant='??', include_primary=True)

        Remarks:
        * since absolute amounts matter, it does not make sense to use relative amounts like fractions
        * mixing entails no order; mixing A and B is equivalent to mixing B and A;
          standard vector encodings do not reflect this
        * the "color groups" were not part of the original dataset; they were added as basic ingredient properties
        * primary ingredient mixtures (A,A) are a special case of binary mixtures (A,B) where A = B
        * a primary ingredient can be described via its position in a vector, or via features, here RGB values;
          the former would not allow prediction of new (or withheld) primary ingredients
        """

        # validate parameters
        # todo: callable equivalent for params, ideally with signature specification
        if isinstance(filter_, str):
            filter_ = params.enumeration(filter_, {"all", "primary", "mixture"})

        # load raw data
        data = self._load_data()

        # filter data
        if filter_ == "all":
            filter_ = lambda e: e
        elif filter_ == "primary":
            filter_ = lambda e: e["type"] == "primary"
        elif filter_ == "mixture":
            filter_ = lambda e: e["type"].startswith("mixture")
        else:  # callable
            pass

        data = tuple(e for e in data if filter_(e))

        # separate labels
        labels = tuple(e["rgb"] for e in data)
        for e in data:
            del e["rgb"]

        # apply post-processing
        data = np.array([samplef(e) for e in data])
        labels = np.array([labelf(e) for e in labels])

        # initialize state
        super().__init__(data=data, labels=labels, **kwargs)


class WatercolorPigments2019DatasetFeatures(Features):
    """Features for the WatercolorPigments2019Dataset."""

    def __init__(self, encoding: str = "one-hot", **kwargs):
        """Initializes features.

        Parameters:
            encoding: how to featurize the two primary pigments in the binary mixtures;
                possible values:
                'one-hot': each primary pigment is a column; for example, 0.02 mL of P3
                    would be encoded as (0, 0, 0.02, 0, ..., 0)
        """

        self._encoding = params.enumeration(encoding, {"one-hot"})

        super().__init__(**kwargs)

    def _apply_one_hot(self, data: WatercolorPigments2019Dataset, primary: dict) -> TabularData:
        """One-hot features."""

        # todo: treat primary features

        dim = 2 * 13
        features = np.zeros((data.num_samples, dim))

        for i, e, y in zip(range(data.num_samples), data.samples(), data.labels()):
            features[i, 0 + e["indexA"] - 1] = e["concentrationA"]
            features[i, 13 + e["indexB"] - 1] = e["concentrationB"]

        labels = np.asarray(data.labels())

        return TabularData(data=features, labels=labels)

    def apply(self, data: TabularData) -> TabularData:
        """Featurize WatercolorPigments2019Dataset."""

        data = params.instance(data, TabularData)

        # set up look-up table for primary pigment data
        primary, primary_data = dict(), WatercolorPigments2019Dataset(filter_="primary")
        for e, rgb in zip(primary_data.samples(), primary_data.labels()):
            primary[e["index"], e["concentration"]] = rgb

        # apply selected encoding
        if self._encoding == "one-hot":
            return self._apply_one_hot(data, primary)
        else:
            raise InvalidParameterError("valid encoding", self._encoding)
