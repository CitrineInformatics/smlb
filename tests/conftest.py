import pytest

from smlb import (
    Data,
    Sampler,
    RandomSubsetSampler,
    RandomVectorSampler,
    VectorSpaceData,
)
from smlb.datasets.synthetic import Friedman1979Data


def _build_sampler(data: Data, max_size: int, rng: int) -> Sampler:
    """Build a sampler from a dataset"""
    num_samples = min(data.num_samples, max_size)
    if isinstance(data, VectorSpaceData):
        return RandomVectorSampler(num_samples, rng=rng)
    else:
        return RandomSubsetSampler(num_samples, rng=rng)


@pytest.fixture(scope='module')
def friedman_1979_data():
    random_state = 0
    num_samples = 500
    dataset = Friedman1979Data(dimensions=5)
    sampler = _build_sampler(dataset, num_samples, random_state)
    return sampler.apply(dataset)
