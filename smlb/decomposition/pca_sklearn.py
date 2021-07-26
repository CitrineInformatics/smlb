from copy import deepcopy
from typing import Optional, Union

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
    def __int_gte_0_f(arg):
        """Test ``arg`` is an integer that is >=0."""
        return params.integer(arg, from_=0)

    @staticmethod
    def __int_gt_0_f(arg):
        """Test ``arg`` is an integer that is >0."""
        return params.integer(arg, from_=1)

    @staticmethod
    def __str_auto_f(arg):
        """Test ``arg == 'auto'``"""
        return params.enumeration(arg, values=("auto",))

    def __init__(
        self,
        rng: Optional[int] = None,
        n_components: Optional[int] = None,
        whiten: bool = False,
        svd_solver: str = "auto",
        tol: float = 0.0,
        iterated_power: Union[int, str] = "auto",
        *args,
        **kwargs
    ):
        """Create a PCA instance.

        Parameters:
            rng: Random state
            n_components: Number of components to keep.
                If ``None``, ``n_components == min(n_samples, n_features) - 1`` when fit.
            whiten : When True (False by default) the ``components_`` vectors are multiplied
                by the square root of n_samples and then divided by the singular values
                to ensure uncorrelated outputs with unit component-wise variances.
                Whitening will remove some information from the transformed signal
                (the relative variance scales of the components) but can sometime
                improve the predictive accuracy of the downstream estimators by
                making their data respect some hard-wired assumptions.
            svd_solver : One of ``{'auto', 'full', 'arpack', 'randomized'}``. Default is ``'auto'``.
                If auto :
                    The solver is selected by a default policy based on ``X.shape`` and
                    ``n_components``: if the input data is larger than 500x500 and the
                    number of components to extract is lower than 80% of the smallest
                    dimension of the data, then the more efficient 'randomized'
                    method is enabled. Otherwise the exact full SVD is computed and
                    optionally truncated afterwards.
                If full :
                    run exact full SVD calling the standard LAPACK solver via
                    ``scipy.linalg.svd`` and select the components by postprocessing
                If arpack :
                    run SVD truncated to n_components calling ARPACK solver via ``scipy.sparse.linalg.svds``.
                    It requires strictly ``0 < n_components < min(X.shape)``
                If randomized :
                    run randomized SVD by the method of Halko et al.
            tol : Tolerance for singular values computed by ``svd_solver == 'arpack'``.
                Must be of range ``[0.0, infinity)``.
            iterated_power : Number of iterations for the power method computed by `svd_solver == 'randomized'`.
                Must be ``'auto'`` or an integer on the range ``[0, infinity)``.
                Default is ``'auto'``
        """

        super().__init__(*args, **kwargs)
        n_components = params.optional_(n_components, self.__int_gt_0_f)
        whiten = params.optional_(whiten, params.boolean)
        svd_solver = params.enumeration(
            svd_solver, values=("auto", "full", "arpack", "randomized")
        )
        tol = params.real(tol, from_=0.0)
        iterated_power = params.any_(iterated_power, self.__int_gte_0_f, self.__str_auto_f)

        self._inverse_transform: bool = False
        self._pca: PCA = PCA(
            n_components=n_components,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=rng,
        )

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
        copy._inverse_transform = not self._inverse_transform
        return copy
