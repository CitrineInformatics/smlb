"""Features

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
2020, Matthias Rupp, Citrine Informatics.

Features transform datasets.
"""


from smlb import DataValuedTransformation, IdentityTransformation


class Features(DataValuedTransformation):
    """Abstract base class for features."""

    # currently adds no functionality, but serves as common base class

    pass


class IdentityFeatures(Features, IdentityTransformation):
    """Leaves dataset unchanged."""

    pass
