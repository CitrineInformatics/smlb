from typing import Callable

from sklearn.feature_selection import SelectPercentile

from smlb import (
    params,
    Data,
    Features,
    TabularData,
)


class SelectPercentileSklearn(Features):
    """Select features based on percentile of highest scores, scikit-learn implementation.

    .. seealso::
        See `here <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile`_ for full documentation.
    """

    def __init__(self, score_func: Callable, percentile: int = 10, *args, **kwargs):
        """Initialize State.

        Parameters:
            score_func: Function that takes two arrays X and y and returns a pair of arrays (scores, pvalues) or a single array with scores.
            percentile: Percent of features to keep. Default is 10.
        """
        score_func = params.callable(score_func, num_pos_or_kw=2)
        percentile = params.integer(percentile, from_=0, to=100)

        super().__init__(*args, **kwargs)
        self._select_percentile = SelectPercentile(score_func=score_func, percentile=percentile)

    def fit(self, data: Data) -> "SelectPercentileSklearn":
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

        self._select_percentile.fit(xtrain, ytrain)

        return self

    def apply(self, data: Data) -> TabularData:
        """Select features from the data.

        Parameters:
            data: data to select features from

        Returns:
            data with selected features
        """
        data = params.instance(data, Data)
        samples = params.real_matrix(data.samples())

        support = self._select_percentile.get_support()
        selected = samples[:, support]

        return TabularData(selected, data.labels())
