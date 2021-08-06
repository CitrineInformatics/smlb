from smlb import (
    params,
    Data,
    Features,
    TabularData,
)
from smlb.feature_selection.selector_protocol_sklearn import SelectorProtocolSklearn


class FeatureSelectorSklearn(Features):
    """Base class for feature selection strategies that use one of scikit-learn's feature selection methods.

    This class relies on a ``selector`` provided on initialization that provides ``fit`` and ``get_support`` methods
     to select features from a dataset.
    """

    def __init__(self, selector: SelectorProtocolSklearn, *args, **kwargs):
        """Initialize state.

        Parameters:
            selector: Feature selection method that provides ``fit`` and ``get_support`` methods.
        """
        super().__init__(*args, **kwargs)
        self._selector: SelectorProtocolSklearn = params.instance(
            selector, SelectorProtocolSklearn
        )

    def fit(self, data: Data) -> "FeatureSelectorSklearn":
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

        self._selector.fit(xtrain, ytrain)

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

        support = self._selector.get_support()
        selected = samples[:, support]

        return TabularData(selected, data.labels())
