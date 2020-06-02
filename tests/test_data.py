"""Dataset tests.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.
"""

import smlb


def validate_data_interface(ds: smlb.Data) -> bool:
    """Tests for compliance with Data interface.

    Runs tests that every Data-compliant class should satisfy.

    Returns:
        True

    Raises:
        AssertionError for failed tests
    """

    # actual or "virtual" abc inheritance
    assert isinstance(ds, smlb.Data)

    if ds.num_samples == float("inf"):
        # infinite data tests
        pass

    else:
        # finite data test

        # integer-representable non-negative size
        assert int(ds.num_samples) == ds.num_samples
        assert ds.num_samples >= 0

        # all samples are returned
        assert len(ds.samples()) == ds.num_samples

        # subsets
        assert ds.subset([]).num_samples == 0
        assert ds.subset().num_samples <= ds.num_samples
        assert ds.subset(duplicates=True).num_samples == ds.num_samples

        # intersection with self
        assert smlb.intersection(ds, ds).num_samples <= ds.num_samples
        # assert smlb.intersection(ds, ds, duplicates=True).num_samples == ds.num_samples  # todo: support this as well

        # complement with self
        assert smlb.complement(ds, ds).num_samples == 0
        # assert smlb.complement(ds, ds, duplicates=True).num_samples == 0  # todo: support this as well

        if ds.is_labeled:
            # all labels are returned
            assert len(ds.labels()) == ds.num_samples

            # subsets
            assert ds.subset([]).is_labeled
            assert ds.subset().is_labeled

            # intersection
            assert smlb.intersection(ds, ds).is_labeled
            # assert smlb.intersection(ds, ds, duplicates=True).is_labeled  # todo: support this as well

            # complement
            assert smlb.complement(ds, ds).is_labeled
            # assert smlb.complement(ds, ds, duplicates=True).is_labeled  # todo: support this as well

    return True
