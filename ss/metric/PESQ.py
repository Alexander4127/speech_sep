import numpy as np
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
import torch
from torch import Tensor

from ss.base.base_metric import BaseMetric
from .util import mask_length


class PESQMetric(BaseMetric):
    def __init__(self, sr: int, *args, **kwargs):
        super().__init__()
        self.pesq = PerceptualEvaluationSpeechQuality(sr, 'nb')

    def __call__(self, short, target, mix_lengths, **kwargs):
        short = mask_length(short, mix_lengths).detach()
        target = target.detach()
        return self.pesq(short, target)
