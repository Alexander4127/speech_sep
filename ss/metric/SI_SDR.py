from ss.base.base_metric import BaseMetric
from .util import si_sdr, mask_length


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, short, target, mix_lengths, **kwargs):
        return si_sdr(mask_length(short, mix_lengths), target)
