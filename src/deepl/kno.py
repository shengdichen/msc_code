"""
implementation based on material of the 2023-spring iteration of
    "Deep Learning in Scientific Computing"
"""

import torch


def parse_int2(x):
    if isinstance(x, (tuple, list)):
        assert len(x) == 2, "x must be a tuple of length 2"
        return int(x[0]), int(x[1])
    else:
        return int(x), int(x)


class SpectralConv2d(torch.nn.Module):
    def __init__(self, in_channel, out_channel, modes=None):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        if modes is None:
            self.modes_x, self.modes_y = None, None
            self.weight_real = torch.nn.Parameter(torch.randn(in_channel, out_channel))
            self.weight_imag = torch.nn.Parameter(torch.randn(in_channel, out_channel))
        else:
            self.modes_x, self.modes_y = parse_int2(modes)
            self.weight_real = torch.nn.Parameter(
                torch.randn(in_channel, out_channel, self.modes_x, self.modes_y)
            )
            self.weight_imag = torch.nn.Parameter(
                torch.randn(in_channel, out_channel, self.modes_x, self.modes_y)
            )

        self.bias_real = torch.nn.Parameter(
            torch.randn(out_channel)[None, :, None, None]
        )
        self.bias_imag = torch.nn.Parameter(
            torch.randn(out_channel)[None, :, None, None]
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(
            self.weight_real, 0, 1 / (self.in_channel * self.out_channel)
        )
        torch.nn.init.uniform_(
            self.weight_imag, 0, 1 / (self.in_channel * self.out_channel)
        )
        torch.nn.init.zeros_(self.bias_real)
        torch.nn.init.zeros_(self.bias_imag)

    def forward(self, x):
        """
        Parameters:
        -----------
            x: torch.Tensor, shape=(batch_size, input_channel, window_size, window_size)
        Returns:
        --------
            y: torch.Tensor, shape=(batch_size, output_channel, window_size, window_size)
        """

        spectral = torch.fft.rfft2(
            x
        )  # spectral [batch_size, input_channel, window_size, window_size//2+1]

        if self.modes_x is None:
            spectral = torch.complex(
                torch.einsum("bixy,io->boxy", spectral.real, self.weight_real)
                + self.bias_real,
                torch.einsum("bixy,io->boxy", spectral.imag, self.weight_imag)
                + self.bias_imag,
            )
            output = torch.fft.irfft2(
                spectral, s=x.shape[-2:]
            )  # output [batch_size, output_channel, window_size, window_size]
        else:
            assert (
                self.modes_y <= (x.shape[-1] // 2 + 1) / 2
            ), f"modes_y must be less than or equal to (window_size // 2 + 1)/2({(x.shape[-1]//2 + 1)/2}), got {self.modes_y}"
            spectral = torch.zeros(
                spectral.shape, dtype=spectral.dtype, device=spectral.device
            )
            B, Ci, H, W = spectral.shape
            Co = self.out_channel
            output_spectral = torch.zeros(
                [B, Co, H, W], dtype=spectral.dtype, device=spectral.device
            )
            output_spectral[:, :, : self.modes_x, : self.modes_y] = torch.complex(
                torch.einsum(
                    "bixy,ioxy->boxy",
                    spectral[:, :, : self.modes_x, : self.modes_y].real,
                    self.weight_real,
                )
                + self.bias_real,
                torch.einsum(
                    "bixy,ioxy->boxy",
                    spectral[:, :, : self.modes_x, : self.modes_y].imag,
                    self.weight_imag,
                )
                + self.bias_imag,
            )
            output_spectral[:, :, -self.modes_x :, -self.modes_y :] = torch.complex(
                torch.einsum(
                    "bixy,ioxy->boxy",
                    spectral[:, :, -self.modes_x :, -self.modes_y :].real,
                    self.weight_real,
                )
                + self.bias_real,
                torch.einsum(
                    "bixy,ioxy->boxy",
                    spectral[:, :, -self.modes_x :, -self.modes_y :].imag,
                    self.weight_imag,
                )
                + self.bias_imag,
            )
            output = torch.fft.irfft2(
                output_spectral, s=x.shape[-2:]
            )  # output [batch_size, output_channel, window_size, window_size]
        return output


class Koopman2d(SpectralConv2d):
    def __init__(self, hidden_channel, modes=None):
        super().__init__(hidden_channel, hidden_channel, modes)


class KNO2d(torch.nn.Module):
    def __init__(
        self, in_channel, out_channel, hidden_channel=64, num_layers=6, modes=4
    ):
        super().__init__()

        self.order = num_layers
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, hidden_channel, kernel_size=1), torch.nn.Tanh()
        )
        self.koopman = Koopman2d(hidden_channel, modes)
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_channel, hidden_channel, kernel_size=1),
            torch.nn.Tanh(),
            torch.nn.Conv2d(hidden_channel, out_channel, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.encoder:
            if isinstance(layer, torch.nn.Conv2d):
                layer.reset_parameters()
        for layer in self.decoder:
            if isinstance(layer, torch.nn.Conv2d):
                layer.reset_parameters()
        self.koopman.reset_parameters()

    def forward(self, x):
        """
        Parameters:
        -----------
            x: torch.Tensor, shape=(batch_size,  channel, H, W)
        Returns:
        --------
            y: torch.Tensor, shape=(batch_size,  channel, H, W)
        """
        x = self.encoder(x)

        skip = x

        for _ in range(self.order):
            x = x + self.koopman(x)

        x = self.decoder(x + skip)

        return x
