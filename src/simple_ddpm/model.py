import torch
import torch.nn as nn
import math


T_EMBEDDING_SIZE = 32


def sinusoidal_positional_embedding(max_seq_len, d_model):
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

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            # nn.SiLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


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

        self.encoder = nn.ModuleList(
            [
                Conv(in_channels + T_EMBEDDING_SIZE, 64),
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
                Conv(2 * 64, out_channels),
            ]
        )
        self.upsamplers = nn.ModuleList(
            [Upsample(512), Upsample(256), Upsample(128), Upsample(64)]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        t_embedding = self.position_embeddings[t]  # (B, T_DIM)
        t_embedding = t_embedding.view(B, T_EMBEDDING_SIZE, 1, 1).expand(
            B, T_EMBEDDING_SIZE, H, W
        )
        x = torch.cat((t_embedding, x), axis=1)

        intermediates = []
        for enc, dow in zip(self.encoder, self.downsamplers):
            x = enc(x)
            intermediates.append(x)  # TODO: Is this a copy?
            x = dow(x)

        x = self.bottleneck(x)

        for dec, up, m in zip(self.decoder, self.upsamplers, reversed(intermediates)):
            x = up(x)
            x = torch.concat((x, m), axis=1)  # Channel dimension
            x = dec(x)

        return x