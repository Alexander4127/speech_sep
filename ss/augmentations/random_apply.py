import random
from typing import Callable, Tuple, Optional

from torch import Tensor


class RandomApply:
    def __init__(self, augmentation: Optional[Callable], p: float):
        assert 0 <= p <= 1
        self.augmentation = augmentation
        self.p = p

    def __call__(self, data: Tensor) -> Tuple[str, Tensor]:
        if self.augmentation is None or random.random() > self.p:
            return '', data
        else:
            return repr(self.augmentation), self.augmentation(data)

    def __repr__(self):
        return f'RandomApply({repr(self.augmentation)})'
