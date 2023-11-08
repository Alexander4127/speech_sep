from torch import distributions, Tensor

from ss.augmentations.base import AugmentationBase


class GaussianNoise(AugmentationBase):
    def __init__(self, sigma=0.01):
        self._noiser = distributions.Normal(loc=0, scale=sigma)

    def __call__(self, data: Tensor):
        return data + self._noiser.sample(data.shape)
