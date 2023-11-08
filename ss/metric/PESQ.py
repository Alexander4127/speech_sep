import numpy as np
from pesq import pesq
import torch
from torch import Tensor

from ss.base.base_metric import BaseMetric
from .util import mask_length


class PESQMetric(BaseMetric):
    def __init__(self, sr: int, *args, **kwargs):
        super().__init__()
        self.sr = sr

    def __call__(self, short, target, mix_lengths, **kwargs):
        short = mask_length(short, mix_lengths).detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        pesq_vals = []
        for short_el, ref_el in zip(short, target):
            pesq_vals.append(pesq(fs=self.sr, ref=ref_el, deg=short_el))
        return np.mean(pesq_vals)
