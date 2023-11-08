import torch
import numpy as np
from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio


def mask_length(x: torch.Tensor, lengths: torch.Tensor):
    x_new = torch.zeros_like(x)
    for idx, length in zip(range(len(lengths)), lengths):
        x_new[idx, :length] = x[idx, :length]
    return x_new


def si_sdr(est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute SI-SDR metric for batch
    :param est: estimated [N, T]
    :param target: real [N, T]
    :return: mean of SI-SDR scores
    """
    return ScaleInvariantSignalDistortionRatio()(est, target)
