import torch
import torch.nn as nn
import math


T_EMBEDDING_SIZE = 32


def sinusoidal_positional_embedding(max_seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Conv, self).__init__()

        self.in_channels = in_channels

        self.t_emb_layer = nn.Sequential(
            nn.SiLU(), nn.Linear(T_EMBEDDING_SIZE, in_channels)
        )

        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                ),
                nn.Sequential(
                    nn.GroupNorm(8, in_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                ),
            ]
        )

        self.sa_norm = nn.GroupNorm(8, in_channels)
        self.sa = nn.MultiheadAttention(in_channels, 4, dropout=0.1, batch_first=True)

        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Input:
            x (torch.Tensor): input of shape (B, C, H, W)
            t_emb (torch.Tensor): embedding time input (B, t_emb)
        """
        B, _, H, W = x.shape
        x_res = self.conv[0](x)
        x_res = x_res + self.t_emb_layer(t_emb).view(B, self.in_channels, 1, 1).expand(
            B, self.in_channels, H, W
        )
        x = x + self.conv[1](x_res)

        C = x.shape[1]
        in_att = self.sa_norm(x.reshape(B, C, -1)).transpose(1, 2)
        out_att, _ = self.sa(in_att, in_att, in_att)
        out_att = out_att.transpose(1, 2).reshape(B, C, H, W)
        x = x + out_att

        x = self.out_conv(x)

        return x


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super(Downsample, self).__init__()

        self.downsample = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, stride=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            channels, channels, kernel_size=2, padding=0, stride=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class UNet(nn.Module):
    def __init__(
        self,
        image_size: int,
        nb_timesteps: int,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super(UNet, self).__init__()

        self.register_buffer(
            "position_embeddings",
            sinusoidal_positional_embedding(nb_timesteps, T_EMBEDDING_SIZE),
        )

        self.preconv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.encoder = nn.ModuleList(
            [
                Conv(64, 64),
                Conv(64, 128),
                Conv(128, 256),
                Conv(256, 512),
            ]
        )
        self.downsamplers = nn.ModuleList(
            [Downsample(64), Downsample(128), Downsample(256), Downsample(512)]
        )
        self.bottleneck = Conv(512, 512)

        self.decoder = nn.ModuleList(
            [
                Conv(2 * 512, 256),
                Conv(2 * 256, 128),
                Conv(2 * 128, 64),
                Conv(2 * 64, 32),
            ]
        )
        self.upsamplers = nn.ModuleList(
            [Upsample(512), Upsample(256), Upsample(128), Upsample(64)]
        )

        self.out_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.preconv(x)

        t_emb = self.position_embeddings[t]  # (B, T_DIM)

        intermediates = []
        for enc, dow in zip(self.encoder, self.downsamplers):
            x = enc(x, t_emb)
            intermediates.append(x)
            x = dow(x)

        x = self.bottleneck(x, t_emb)

        for dec, up, m in zip(self.decoder, self.upsamplers, reversed(intermediates)):
            x = up(x)
            x = torch.concat((x, m), axis=1)  # Channel dimension
            x = dec(x, t_emb)

        x = self.out_conv(x)

        return x
