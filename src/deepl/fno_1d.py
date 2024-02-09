import torch


class SpectralConv1d(torch.nn.Module):
    def __init__(self, n_channels_lhs: int, n_channels_rhs: int, n_modes: int):
        super().__init__()

        self._n_channels_in = n_channels_lhs
        self._n_channels_out = n_channels_rhs
        self._n_modes_1 = n_modes

        self._scale = 1 / (n_channels_lhs * n_channels_rhs)
        self._weights_1 = torch.nn.Parameter(
            self._scale
            * torch.rand(
                (self._n_channels_in, self._n_channels_out, self._n_modes_1),
                dtype=torch.cfloat,
            )
        )

    def forward(self, lhss: torch.Tensor) -> torch.Tensor:
        # NOTE:
        # lhss.shape == [batch_size, n_channels_lhs, n_gridpts]

        batchsize = lhss.shape[0]

        # Fourier coeffcients up to factor of e^(- SOME_CONST)
        lhss_fourier = torch.fft.rfft(lhss)

        # apply the required Fourier modes (first n modes)
        res = torch.zeros(
            (batchsize, self._n_channels_out, lhss.size(-1) // 2 + 1),
            device=lhss.device,
            dtype=torch.cfloat,
        )
        res[:, :, : self._n_modes_1] = self._mult_complex_1d(
            lhss_fourier[:, :, : self._n_modes_1], self._weights_1
        )

        # [batch_size, out_channels, n_gridpts]
        return torch.fft.irfft(res, n=lhss.size(-1))

    def _mult_complex_1d(self, lhss, weights) -> torch.Tensor:
        """
        lhss: (batchsize, n_channels_lhs, n_gridpts_x1)
        weights: (n_channels_lhs, n_channels_rhs, n_gridpts_x1)
        rhss(return): (batchsize, n_channels_rhs, n_gridpts_x1)
        """
        return torch.einsum("bix,iox->box", lhss, weights)


class FNO1d(torch.nn.Module):
    def __init__(
        self,
        n_channels_lhs: int = 2,
        n_channels_fourier: int = 64,
        n_modes_fourier: int = 16,
        n_channels_rhs: int = 1,
    ):
        """
        e.g., lhs := [f(x1, x2), x1, x2] <=> n_channels_lhs = 3
        """

        super().__init__()

        self._linear_p = torch.nn.Linear(n_channels_lhs, n_channels_fourier)

        self._spect_0 = SpectralConv1d(
            n_channels_fourier, n_channels_fourier, n_modes_fourier
        )
        self._spect_1 = SpectralConv1d(
            n_channels_fourier, n_channels_fourier, n_modes_fourier
        )
        self._spect_2 = SpectralConv1d(
            n_channels_fourier, n_channels_fourier, n_modes_fourier
        )
        self._conv_0 = torch.nn.Conv1d(n_channels_fourier, n_channels_fourier, 1)
        self._conv_1 = torch.nn.Conv1d(n_channels_fourier, n_channels_fourier, 1)
        self._conv_2 = torch.nn.Conv1d(n_channels_fourier, n_channels_fourier, 1)

        size_pre_out = 32
        self._linear_q = torch.nn.Linear(n_channels_fourier, size_pre_out)
        self._layer_out = torch.nn.Linear(size_pre_out, n_channels_rhs)

        self._activation = torch.nn.Tanh()
        self._padding_if_nonperiodic = 1  # pad the domain if input is non-periodic

    def forward(self, lhss: torch.Tensor) -> torch.Tensor:
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)

        # lhss: 10 (batch-size); each lhss (1001, 2)
        rhss = self._linear_p(lhss)  # 10, 1001, 64

        rhss = rhss.permute(0, 2, 1)  # 10, 64, 1001
        # x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic
        rhss = self._apply_layer_fourier(rhss, self._spect_0, self._conv_0)  # unchanged
        rhss = self._apply_layer_fourier(rhss, self._spect_1, self._conv_1)  # unchanged
        rhss = self._apply_layer_fourier(rhss, self._spect_2, self._conv_2)  # unchanged
        # x = x[..., :-self.padding]  # pad the domain if input is non-periodic
        rhss = rhss.permute(0, 2, 1)  # 10, 1001, 64

        # -linear_q-> 10, 1001, 32 -linear_out-> # 10, 1001, 1
        return self._layer_out(self._activation(self._linear_q(rhss)))

    def _apply_layer_fourier(
        self,
        lhss: torch.Tensor,
        layer_spectral: torch.nn.Module,
        layer_conv: torch.nn.Module,
    ) -> torch.Tensor:
        return self._activation(layer_spectral(lhss) + layer_conv(lhss))
