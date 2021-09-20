from copy import deepcopy

from sklearn.preprocessing import StandardScaler

from smlb import (
    DataValuedTransformation,
    InvertibleTransformation,
    Data,
    TabularData,
    params,
)


class StandardScalarSklearn(DataValuedTransformation, InvertibleTransformation):
    """Standardize features by removing the mean and scaling to unit variance, scikit-learn implementation.

    The standard score of a sample ``x`` is calculated as ``z = (x - u) / s``,
    where ``u`` is the mean of the training samples or zero if ``with_mean=False``, and
    ``s`` is the standard deviation of the training samples or one if ``with_std=False``.

    .. seealso::
        See `here <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler>`_ for full documentation.
    """

    def __init__(
        self, copy: bool = True, with_mean: bool = True, with_std: bool = True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._inverse_transform: bool = False
        self._standard_scalar = StandardScaler(
            copy=params.boolean(copy),
            with_mean=params.boolean(with_mean),
            with_std=params.boolean(with_std),
        )

    def fit(self, data: Data) -> "StandardScalarSklearn":
        """Fit the model with input ``data``.

        Parameters:
            data: data to fit

        Returns:
            the instance itself
        """
        data = params.instance(data, Data)
        xtrain = params.real_matrix(data.samples(), nrows=data.num_samples)
        self._standard_scalar.fit(xtrain)
        return self

    def apply(self, data: Data) -> TabularData:
        """Transform the data.

        Parameters:
            data: data to transform

        Returns:
            transformed data
        """
        data = params.instance(data, Data)
        xpred = params.real_matrix(data.samples())

        if self._inverse_transform:
            preds = self._standard_scalar.inverse_transform(xpred)
        else:
            preds = self._standard_scalar.transform(xpred)

        return TabularData(preds, data.labels())

    def inverse(self) -> "StandardScalarSklearn":
        """Return inverse standardization if this object is a forward transform or the reverse if this object is an inverse transform.

        Inverse transforms standardized data back to the original space.
        """
        copy = deepcopy(self)
        copy._inverse_transform = not self._inverse_transform
        return copy
