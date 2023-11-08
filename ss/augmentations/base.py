from torch import Tensor


class AugmentationBase:
    def __call__(self, data: Tensor) -> Tensor:
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}"
