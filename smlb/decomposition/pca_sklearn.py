from copy import deepcopy
from typing import Optional

from sklearn.decomposition import PCA

from smlb import (
    DataValuedTransformation,
    InvertibleTransformation,
    Data,
    TabularData,
    params,
)


class PCASklearn(DataValuedTransformation, InvertibleTransformation):
    """Principal component analysis (PCA), scikit-learn implementation.

    .. seealso::
        See `here <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA>`_ for full documentation.
    """

    @staticmethod
    def __nneg_int_f(arg):
        """Test ``arg`` is an integer that is >0."""
        return params.integer(arg, from_=1)

    def __init__(
        self,
        rng: Optional[int] = None,
        n_components: Optional[int] = None,
        inverse_transform: bool = False,
        *args,
        **kwargs
    ):
        """Create a PCA instance.

        Parameters:
            rng: Random state
            n_components: Number of components to keep.
                          If ``None``, ``n_components == min(n_samples, n_features) - 1`` when fit.
            inverse_transform: Whether to perform inverse transformation. Default is ``False``
        """

        super().__init__(*args, **kwargs)
        n_components = params.optional_(n_components, self.__nneg_int_f)

        self._inverse_transform: bool = params.boolean(inverse_transform)
        self._pca: PCA = PCA(n_components, random_state=rng)

    def fit(self, data: Data) -> "PCASklearn":
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

        self._pca.fit(xtrain, ytrain)

        return self

    def apply(self, data: Data) -> TabularData:
        """Apply PCA to the data.

        Parameters:
            data: data to transform

        Returns:
            transformed data
        """
        data = params.instance(data, Data)
        xpred = params.real_matrix(data.samples())

        if self._inverse_transform:
            preds = self._pca.inverse_transform(xpred)
        else:
            preds = self._pca.transform(xpred)

        return TabularData(preds, data.labels())

    def inverse(self) -> "PCASklearn":
        """Return inverse PCA.

        Inverse PCA transforms from the reduced components back to the original space.
        """
        copy = deepcopy(self)
        copy._inverse_transform = not copy._inverse_transform
        return copy
