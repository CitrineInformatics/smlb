from copy import deepcopy
from typing import Optional

import prince

from smlb import (
    DataValuedTransformation,
    InvertibleTransformation,
    Data,
    TabularData,
    params,
)


class PCAPrince(DataValuedTransformation, InvertibleTransformation):
    """Principal component analysis (PCA), prince implementation.

    .. seealso::
        See `here <https://github.com/MaxHalford/prince#principal-component-analysis-pca>`_ for full documentation.
    """

    def __init__(
        self,
        rng: Optional[int] = None,
        rescale_with_mean: bool = True,
        rescale_with_std: bool = True,
        n_components: int = 2,
        n_iter: int = 3,
        *args,
        **kwargs
    ):
        """Initialize state.

        Parameters:
            rng: Random state
            rescale_with_mean: Whether to subtract each column's mean or not.
            rescale_with_std: Whether to divide each column by it's standard deviation or not.
            n_components: The number of principal components to compute.
            n_iter: The number of iterations used for computing the SVD.
        """

        super().__init__(*args, **kwargs)
        rescale_with_mean = params.boolean(rescale_with_mean)
        rescale_with_std = params.boolean(rescale_with_std)
        n_components = params.integer(n_components, from_=1)
        n_iter = params.integer(n_iter, from_=1)

        self._inverse_transform: bool = False
        self._pca: prince.PCA = prince.PCA(
            rescale_with_mean=rescale_with_mean,
            rescale_with_std=rescale_with_std,
            n_components=n_components,
            n_iter=n_iter,
            random_state=rng,
            as_array=True,
        )

    def fit(self, data: Data) -> "PCAPrince":
        """Fit the model with input ``data``.

        Parameters:
            data: data to fit

        Returns:
            the instance itself
        """
        data = params.instance(data, Data)

        xtrain = params.real_matrix(data.samples(), nrows=data.num_samples)
        ytrain = params.real_vector(data.labels(), dimensions=data.num_samples)

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

    def inverse(self) -> "PCAPrince":
        """Return inverse PCA.

        Inverse PCA transforms from the reduced components back to the original space.
        """
        copy = deepcopy(self)
        copy._inverse_transform = not self._inverse_transform
        return copy
