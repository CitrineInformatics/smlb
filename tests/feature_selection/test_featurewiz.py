from smlb import Data
from smlb.feature_selection import Featurewiz


def test_transform(friedman_1979_data: Data):
    featurewiz = Featurewiz()
    featurewiz.fit(friedman_1979_data)
    selected = featurewiz.apply(friedman_1979_data)
    # all features from input data should be selected since no low-information variables are present
    assert selected.samples().shape == (friedman_1979_data.num_samples, 5)
