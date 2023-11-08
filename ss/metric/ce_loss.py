import torch
from torch import Tensor

from ss.base.base_metric import BaseMetric
from .util import mask_length


class CELossMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ce = torch.nn.CrossEntropyLoss()

    def __call__(self, speaker_probs, speaker_id, **kwargs):
        return self.ce(speaker_probs, speaker_id)
