"""
implementation based on material of Tutorial-10 of the 2023-spring iteration of
    "Deep Learning in Scientific Computing"
with modifications for readability
"""

import torch

from src.util import dataset as dataset_util


class SpectralConv2d(torch.nn.Module):
    def __init__(
        self,
        n_channels_lhs: int,
        n_modes_1: int,
        n_modes_2: int,
        n_channels_rhs: int,
    ):
        super().__init__()

        self._n_channels_lhs, self._n_channels_rhs = n_channels_lhs, n_channels_rhs
        self._n_modes_1, self._n_modes_2 = n_modes_1, n_modes_2

        scale = 1 / (n_channels_lhs * n_channels_rhs)
        self._weights_1 = torch.nn.Parameter(
            scale
            * torch.rand(
                (n_channels_lhs, n_channels_rhs, self._n_modes_1, self._n_modes_2),
                dtype=torch.cfloat,
            )
        )
        self._weights_2 = torch.nn.Parameter(
            scale
            * torch.rand(
                (n_channels_lhs, n_channels_rhs, self._n_modes_1, self._n_modes_2),
                dtype=torch.cfloat,
            )
        )

    def forward(self, lhss: torch.Tensor) -> torch.Tensor:
        batchsize = lhss.shape[0]

        lhss_fourier = torch.fft.rfft2(lhss)

        res = torch.zeros(
            (batchsize, self._n_channels_rhs, lhss.size(-2), lhss.size(-1) // 2 + 1),
            dtype=torch.cfloat,
            device=lhss.device,
        )
        res[:, :, : self._n_modes_1, : self._n_modes_2] = self._multiply_complex_2d(
            lhss_fourier[:, :, : self._n_modes_1, : self._n_modes_2], self._weights_1
        )
        res[:, :, -self._n_modes_1 :, : self._n_modes_2] = self._multiply_complex_2d(
            lhss_fourier[:, :, -self._n_modes_1 :, : self._n_modes_2], self._weights_2
        )

        return torch.fft.irfft2(res, s=(lhss.size(-2), lhss.size(-1)))

    def _multiply_complex_2d(
        self, lhss: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """
        lhss: (batchsize, n_channels_lhs, n_gridpts_x1, n_gridpts_x2)
        weights: (n_channels_lhs, n_channels_rhs, n_gridpts_x1, n_gridpts_x2)
        rhss(return): (batchsize, n_channels_rhs, n_gridpts_x1, n_gridpts_x2)
        """
        return torch.einsum("bixy,ioxy->boxy", lhss, weights)


class FNO2d(torch.nn.Module):
    def __init__(
        self,
        n_channels_lhs: int = 3,
        n_channels_fourier: int = 64,
        n_modes_fourier_x1: int = 16,
        n_modes_fourier_x2: int = 16,
        n_layers_fourier: int = 3,
        n_channels_rhs: int = 1,
        activation=torch.nn.functional.gelu,
        padding_frac=1 / 4,
    ):
        """
        e.g., lhs := [f(x1, x2), x1, x2] <=> n_channels_lhs = 3
        """

        super().__init__()

        self._layer_in = torch.nn.Linear(n_channels_lhs, n_channels_fourier)

        self._n_layers = n_layers_fourier
        self._convolutions = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(n_channels_fourier, n_channels_fourier, 1)
                for __ in range(self._n_layers)
            ]
        )
        self._spectrals = torch.nn.ModuleList(
            [
                SpectralConv2d(
                    n_channels_lhs=n_channels_fourier,
                    n_modes_1=n_modes_fourier_x1,
                    n_modes_2=n_modes_fourier_x2,
                    n_channels_rhs=n_channels_fourier,
                )
                for __ in range(self._n_layers)
            ]
        )

        size_pre_out = 128
        self._layer_out_pre = torch.nn.Linear(n_channels_fourier, size_pre_out)
        self._layer_out = torch.nn.Linear(size_pre_out, n_channels_rhs)

        self._activation = activation

        # pad the domain if input is non-periodic
        self._padding_frac = padding_frac

    def forward(self, lhss: torch.Tensor) -> torch.Tensor:
        """
        lhss: (batch_size, n_channels_lhs, n_gridpts_x1, n_gridpts_x2)
        return: (batch_size, n_channels_rhs, n_gridpts_x1, n_gridpts_x2)
        """

        rhss = self._forward_pre(lhss)

        padding_x1 = int(round(rhss.shape[-1] * self._padding_frac))
        padding_x2 = int(round(rhss.shape[-2] * self._padding_frac))
        rhss = torch.nn.functional.pad(rhss, [0, padding_x1, 0, padding_x2])

        for i, (spect, conv) in enumerate(zip(self._spectrals, self._convolutions)):
            rhss = spect(rhss) + conv(rhss)
            if i != self._n_layers - 1:
                rhss = self._activation(rhss)
        rhss = rhss[..., :-padding_x1, :-padding_x2]

        return self._forward_post(rhss)

    def _forward_pre(self, tensor: torch.Tensor) -> torch.Tensor:
        # (batch_size, x1, x2, n_channels_lhs)
        tensor = dataset_util.Reorderer.components_to_last_tensors(tensor)[0]
        # (batch_size, x1, x2, n_channels_fourier)
        tensor = self._layer_in(tensor)
        # (batch_size, n_channels_fourier, x1, x2)
        tensor = dataset_util.Reorderer.components_to_second_tensors(tensor)[0]
        return tensor

    def _forward_post(self, tensor: torch.Tensor) -> torch.Tensor:
        # (batch_size, x1, x2, n_channels_fourier)
        tensor = dataset_util.Reorderer.components_to_last_tensors(tensor)[0]
        # (batch_size, x1, x2, size_pre_out)
        tensor = self._layer_out_pre(tensor)
        # (batch_size, x1, x2, n_channels_rhs)
        tensor = self._layer_out(self._activation(tensor))

        # (batch_size, n_channels_rhs, x1, x2)
        tensor = dataset_util.Reorderer.components_to_second_tensors(tensor)[0]
        return tensor
