from typing import Callable

from sklearn.feature_selection import SelectPercentile

from smlb import params
from smlb.feature_selection.feature_selector_sklearn import FeatureSelectorSklearn


class SelectPercentileSklearn(FeatureSelectorSklearn):
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
        selector = SelectPercentile(score_func=score_func, percentile=percentile)
        super().__init__(selector=selector, *args, **kwargs)
