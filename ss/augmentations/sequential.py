from typing import List, Callable

from torch import Tensor

from ss.augmentations.base import AugmentationBase
from ss.augmentations.random_apply import RandomApply


class SequentialAugmentation(AugmentationBase):
    def __init__(self, augmentation_list: List[Callable]):
        self.augmentation_list = augmentation_list
        self._description: str

    def __call__(self, data: Tensor) -> Tensor:
        x = data
        self._description = ""
        for idx, augmentation in enumerate(self.augmentation_list):
            desc = repr(x)
            if isinstance(augmentation, RandomApply):
                desc, x = augmentation(x)
            else:
                x = augmentation(x)
            self._description += f' {desc}' if desc else ''
        return x

    def __repr__(self):
        return self._description
