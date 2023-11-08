from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from ss.base.base_metric import BaseMetric
from .util import mask_length


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio()

    def __call__(self, short, target, mix_lengths, **kwargs):
        return self.si_sdr(mask_length(short, mix_lengths).cpu(), target.cpu())
