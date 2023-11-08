import logging

import torch
import torch.nn as nn

from .si_sdr_loss import SISDRLoss


class CompositeLoss(nn.Module):
    def __init__(self, lam, sdr_params, ce_params, **kwargs):
        """
        Construct composite loss from SI-SDR and CE
        :param lam: coefficient before CE-Loss
        :param sdr_params: params for SI-SDR-Loss
        :param ce_params: params for CE-Loss
        :param kwargs: other params
        """
        super().__init__(**kwargs)
        self._sdr_loss = SISDRLoss(**sdr_params)
        self._ce_loss = torch.nn.CrossEntropyLoss(**ce_params)
        self.lam = lam

    def __call__(self, speaker_probs, speaker_id, **kwargs):
        sdr_loss = self._sdr_loss(**kwargs)
        ce_loss = self._ce_loss(speaker_probs, speaker_id)
        return sdr_loss + self.lam * ce_loss
