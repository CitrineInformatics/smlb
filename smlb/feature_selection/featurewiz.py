from typing import Optional, Sequence, Set, Union

from featurewiz import featurewiz
import numpy as np
import pandas as pd

from smlb import (
    params,
    Data,
    Features,
    NotFittedError,
    TabularData,
)


class Featurewiz(Features):
    """Select features using featurewiz.

    .. seealso::
        See `here <https://github.com/AutoViML/featurewiz>`_ for full documentation.
    """

    FEATURE_ENGG_OPTIONS: Set[str] = {"", "interactions", "groupby", "target"}
    CATEGORY_ENCODER_OPTIONS: Set[str] = {
        "",
        "HashingEncoder",
        "SumEncoder",
        "PolynomialEncoder",
        "BackwardDifferenceEncoder",
        "OneHotEncoder",
        "HelmertEncoder",
        "OrdinalEncoder",
        "FrequencyEncoder",
        "BaseNEncoder",
        "TargetEncoder",
        "CatBoostEncoder",
        "WOEEncoder",
        "JamesSteinEncoder",
    }

    def __init__(
        self,
        corr_limit: float = 0.7,
        verbose: int = 0,
        feature_engg: Union[str, Sequence[str]] = "",
        category_encoders: Union[str, Sequence[str]] = "",
        *args,
        **kwargs,
    ):
        self._corr_limit = params.real(corr_limit, from_=0.0, to=1.0)
        self._verbose = params.integer(verbose, from_=0, to=2)

        is_valid_feature_engg = lambda x: params.enumeration(x, self.FEATURE_ENGG_OPTIONS)
        is_seq_feature_engg = lambda x: params.sequence(x, testf=is_valid_feature_engg)
        self._feature_engg = params.any_(feature_engg, is_valid_feature_engg, is_seq_feature_engg)

        is_valid_category_encoders = lambda x: params.enumeration(x, self.CATEGORY_ENCODER_OPTIONS)
        is_seq_category_encoders = lambda x: params.sequence(x, testf=is_valid_category_encoders)
        self._category_encoders = params.any_(
            category_encoders, is_valid_category_encoders, is_seq_category_encoders
        )

        self._support: Optional[Sequence[int]] = None

        super().__init__(*args, **kwargs)

    def fit(self, data: Data) -> "Featurewiz":
        """Fit the model with input ``data``.

        Parameters:
            data: data to fit

        Returns:
            the instance itself
        """
        data = params.instance(data, Data)
        n = data.num_samples

        xtrain = params.real_matrix(data.samples(), nrows=n)
        ytrain = params.real_vector(data.labels(), dimensions=n)

        n_features = xtrain.shape[1]
        columns = [str(i) for i in range(n_features)] + ["target"]
        data = np.hstack((xtrain, ytrain[:, np.newaxis]))
        df = pd.DataFrame(data=data, columns=columns)

        features, _ = featurewiz(
            dataname=df,
            target="target",
            corr_limit=self._corr_limit,
            verbose=self._verbose,
            feature_engg=self._feature_engg,
            category_encoders=self._category_encoders,
        )
        self._support = sorted([int(i) for i in features])
        return self

    def apply(self, data: Data) -> TabularData:
        """Select features from the data.

        Parameters:
            data: data to select features from

        Returns:
            data with selected features
        """
        if self._support is None:
            raise NotFittedError(self)

        data = params.instance(data, Data)
        samples = params.real_matrix(data.samples())

        selected = samples[:, self._support]

        return TabularData(selected, data.labels())
