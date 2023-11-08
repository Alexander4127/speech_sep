from itertools import chain
import logging
import typing as tp

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np

from ss.base import BaseModel
from ss.logger.logger import logger
from .cnn import TCNBlock, TCNBlockSpeaker, ResBlock


class SpexPlusModel(nn.Module):
    def __init__(self,
                 tcn_block_params: dict,
                 res_block_out: int,
                 speaker_dim: int,
                 num_speakers: int,
                 enc_dim: int,
                 L1: int,
                 L2: int,
                 L3: int
                 ):
        super().__init__()

        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.stride = L1 // 2
        self.enc_dim = enc_dim

        # encoders
        self.short_encoder = nn.Conv1d(in_channels=1, out_channels=enc_dim, kernel_size=L1, stride=self.stride)
        self.middle_encoder = nn.Conv1d(in_channels=1, out_channels=enc_dim, kernel_size=L2, stride=self.stride)
        self.long_encoder = nn.Conv1d(in_channels=1, out_channels=enc_dim, kernel_size=L3, stride=self.stride)

        # layer norm and projection (after concatenation)
        out_channels = tcn_block_params['in_channels']
        self.out_chan = out_channels
        self.layer_norm = nn.LayerNorm(normalized_shape=3 * enc_dim)
        self.projection = nn.Conv1d(in_channels=3 * enc_dim, out_channels=out_channels, kernel_size=1)

        # TCN Blocks
        self.block1_spk = TCNBlockSpeaker(**tcn_block_params, speaker_dim=speaker_dim)
        self.block1_ord = self._make_pyramid(**tcn_block_params)
        self.block2_spk = TCNBlockSpeaker(**tcn_block_params, speaker_dim=speaker_dim)
        self.block2_ord = self._make_pyramid(**tcn_block_params)
        self.block3_spk = TCNBlockSpeaker(**tcn_block_params, speaker_dim=speaker_dim)
        self.block3_ord = self._make_pyramid(**tcn_block_params)
        self.block4_spk = TCNBlockSpeaker(**tcn_block_params, speaker_dim=speaker_dim)
        self.block4_ord = self._make_pyramid(**tcn_block_params)

        # masks
        self.mask1 = nn.Conv1d(in_channels=out_channels, out_channels=enc_dim, kernel_size=1)
        self.mask2 = nn.Conv1d(in_channels=out_channels, out_channels=enc_dim, kernel_size=1)
        self.mask3 = nn.Conv1d(in_channels=out_channels, out_channels=enc_dim, kernel_size=1)

        # decoders
        self.short_decoder = nn.ConvTranspose1d(in_channels=enc_dim, out_channels=1, kernel_size=L1, stride=self.stride)
        self.middle_decoder = nn.ConvTranspose1d(in_channels=enc_dim, out_channels=1, kernel_size=L2, stride=self.stride)
        self.long_decoder = nn.ConvTranspose1d(in_channels=enc_dim, out_channels=1, kernel_size=L3, stride=self.stride)

        # speaker embeddings
        self.speaker_layer_norm = nn.LayerNorm(normalized_shape=3 * enc_dim)

        self.speaker_encoder = nn.Sequential(
            nn.Conv1d(in_channels=3 * enc_dim, out_channels=out_channels, kernel_size=1),
            ResBlock(in_channels=out_channels, out_channels=out_channels),
            ResBlock(in_channels=out_channels, out_channels=res_block_out),
            ResBlock(in_channels=res_block_out, out_channels=res_block_out),
            nn.Conv1d(in_channels=res_block_out, out_channels=speaker_dim, kernel_size=1)
        )

        # classifier head
        self.head = nn.Linear(speaker_dim, num_speakers)

    @staticmethod
    def _make_pyramid(num_blocks, **tcn_block_params):
        return nn.Sequential(*[TCNBlock(**tcn_block_params, dilation=2**b) for b in range(1, num_blocks)])

    def _encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, ...]:
        w1 = F.relu(self.short_encoder(x.unsqueeze(1)))

        T = w1.shape[-1]
        pad_middle = (T - 1) * self.stride + self.L2 - x.shape[-1]
        pad_long = (T - 1) * self.stride + self.L3 - x.shape[-1]

        w2 = F.relu(self.middle_encoder(F.pad(x, [0, pad_middle], "constant", 0).unsqueeze(1)))
        w3 = F.relu(self.long_encoder(F.pad(x, [0, pad_long], "constant", 0).unsqueeze(1)))

        assert w1.shape[1] == w2.shape[1] == w3.shape[1] == self.enc_dim

        return w1, w2, w3, torch.cat([w1, w2, w3], dim=1)  # [N x 3*enc_dim x T]

    def forward(self, mix: torch.Tensor, target: torch.Tensor, ref: torch.Tensor, **kwargs):
        assert len(mix.shape) == len(target.shape) == len(ref.shape) == 2, \
            f'All {mix.shape, target.shape, ref.shape} are expected to have length = 2'

        # encoding of mixed audio
        len_mix = mix.shape[-1]
        w1, w2, w3, mix_enc = self._encode(mix)
        mix_proj = self.projection(self.layer_norm(mix_enc.transpose(1, 2)).transpose(1, 2))  # [N, out_chan, T]

        # encoding of speaker
        len_spk = ref.shape[-1]
        _, _, _, spk_enc = self._encode(ref)
        spk_proj = self.speaker_layer_norm(spk_enc.transpose(1, 2)).transpose(1, 2)
        spk_proj = self.speaker_encoder(spk_proj)

        out_len_spk = (len_spk - self.L1) // self.stride + 1  # encoder block
        out_len_spk //= 27  # max pooling in 3 res blocks
        spk_proj = torch.sum(spk_proj, dim=-1) / float(out_len_spk)

        out = self.block1_spk(mix_proj, spk_proj)
        out = self.block1_ord(out)
        out = self.block2_spk(out, spk_proj)
        out = self.block2_ord(out)
        out = self.block3_spk(out, spk_proj)
        out = self.block3_ord(out)
        out = self.block4_spk(out, spk_proj)
        out = self.block4_ord(out)

        mask1 = F.relu(self.mask1(out))
        mask2 = F.relu(self.mask2(out))
        mask3 = F.relu(self.mask3(out))

        short = self.short_decoder(w1 * mask1).squeeze(1)
        short = F.pad(short, [0, len_mix - short.shape[-1]], "constant", 0)
        middle = self.middle_decoder(w2 * mask2).squeeze(1)[:, :len_mix]
        long = self.long_decoder(w3 * mask3).squeeze(1)[:, :len_mix]

        assert short.shape[-1] == middle.shape[-1] == long.shape[-1] == len_mix

        return {"short": short, "middle": middle, "long": long, "speaker_probs": self.head(spk_proj)}

    def __repr__(self):
        return f'{super().__repr__()}\n\nNum of parameters: {sum(p.numel() for p in self.parameters())}'
