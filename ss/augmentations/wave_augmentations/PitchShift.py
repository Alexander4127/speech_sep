import torch_audiomentations
from torch import Tensor

from ss.augmentations.base import AugmentationBase


class PitchShift(AugmentationBase):
    def __init__(self, sample_rate):
        self._aug = torch_audiomentations.PitchShift(sample_rate=sample_rate)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
