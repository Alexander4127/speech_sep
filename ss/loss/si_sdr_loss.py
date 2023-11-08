import torch
from torch import Tensor
import torch.nn as nn

from numpy import isclose
from ss.metric.util import si_sdr, mask_length


class SISDRLoss(nn.Module):
    def __init__(self, sh: float = 0.8, mid: float = 0.1, lng: float = 0.1, **kwargs):
        """
        Initialize weights for 3 types of decoders
        :param sh: for short decoder
        :param mid: for middle decoder
        :param lng: for long decoder
        """
        super().__init__(**kwargs)
        assert isclose(sh + mid + lng, 1.0), f'Sum of coefficients {sh, mid, lng} is different from 1.'
        self.sh = sh
        self.mid = mid
        self.lng = lng

    @staticmethod
    def _compute_si_sdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        pred = pred - pred.mean(pred, dim=-1, keepdim=True)
        target = target - pred.mean(target, dim=-1, keepdim=True)
        s = torch.sum(pred * target, dim=-1, keepdim=True) * target / (torch.norm(target, keepdim=True)**2 + eps)
        return 20 * torch.log10(eps + torch.norm(s, dim=-1) / (torch.norm(pred - s) + eps))

    def forward(self, short, middle, long, target, mix_lengths, **batch) -> Tensor:
        sdr_short = self._compute_si_sdr(mask_length(short, mix_lengths), target)
        sdr_middle = self._compute_si_sdr(mask_length(middle, mix_lengths), target)
        sdr_long = self._compute_si_sdr(mask_length(long, mix_lengths), target)
        return - (self.sh * sdr_short + self.mid * sdr_middle + self.lng * sdr_long)
