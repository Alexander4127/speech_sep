from typing import List

import torch
from torch import Tensor

from ss.base.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, speaker_probs: Tensor, speaker_id: Tensor, **kwargs):
        pred_id = torch.argmax(speaker_probs, dim=-1)
        assert len(speaker_id) == len(pred_id)
        return torch.mean((speaker_id == pred_id).to(torch.float32)).item()
