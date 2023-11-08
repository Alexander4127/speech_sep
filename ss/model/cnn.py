import torch
import torch.nn as nn

from .norm import GLayerNorm


class TCNBase(nn.Module):
    def __init__(self, in_channels, out_channels, conv_channels, kernel_size, dilation, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=conv_channels, kernel_size=1)

        self.p_relu1 = nn.PReLU()
        self.g_norm1 = GLayerNorm(n_channels=conv_channels)

        self.d_conv = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            groups=conv_channels,
            padding=(dilation * (kernel_size - 1)) // 2,
            dilation=dilation
        )
        self.p_relu2 = nn.PReLU()

        self.g_norm2 = GLayerNorm(n_channels=conv_channels)

        self.conv3 = nn.Conv1d(in_channels=conv_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 3, 'Shape should be [N, C, T]'
        out = self.conv1(x)
        out = self.g_norm1(self.p_relu1(out))
        out = self.d_conv(out)
        out = self.g_norm2(self.p_relu2(out))
        out = self.conv3(out)
        return out


class TCNBlock(TCNBase):
    def __init__(self, in_channels, conv_channels, kernel_size, dilation=1, **kwargs):
        super().__init__(in_channels, in_channels, conv_channels, kernel_size, dilation)

    def forward(self, x: torch.Tensor):
        out = super().forward(x)
        return out + x


class TCNBlockSpeaker(TCNBase):
    def __init__(self, in_channels, speaker_dim, conv_channels, kernel_size=3, dilation=1, **kwargs):
        super().__init__(in_channels + speaker_dim, in_channels, conv_channels, kernel_size, dilation)

    def forward(self, x: torch.Tensor, spx: torch.Tensor):
        assert len(x.shape) == 3 and len(spx.shape) == 2, f'Expected {x.shape} = [N, C, T]. {spx.shape} == [N, D]'
        spx = torch.unsqueeze(spx, -1)
        spx = spx.repeat(1, 1, x.shape[-1])

        x_new = torch.cat([x, spx], dim=1)
        out = super().forward(x_new)

        return out + x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.p_relu1 = nn.PReLU()
        self.batch_norm1 = nn.BatchNorm1d(num_features=out_channels)

        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(num_features=out_channels)

        self.downsampler = None if in_channels == out_channels else \
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)

        self.p_relu2 = nn.PReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, x):
        assert len(x.shape) == 3, f'Expected {x.shape} = [N, C, T].'
        out = self.conv1(x)
        out = self.p_relu1(self.batch_norm1(out))
        out = self.conv2(out)
        out = self.batch_norm2(out)

        out = out + (x if self.downsampler is None else self.downsampler(x))
        out = self.p_relu2(out)
        out = self.max_pool(out)

        return out
