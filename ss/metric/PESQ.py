import numpy as np
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
import torch
from torch import Tensor

from ss.base.base_metric import BaseMetric
from .util import mask_length


class PESQMetric(BaseMetric):
    def __init__(self, sr: int = 16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(sr, 'wb')

    def __call__(self, short, target, mix_lengths, **kwargs):
        short = mask_length(short, mix_lengths).detach().cpu()
        target = target.detach().cpu()
        return self.pesq(short, target)
