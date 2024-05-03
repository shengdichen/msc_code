import typing

from neuralop import models as fno

from src.deepl import cno
from src.deepl import fno_2d as fno_ours
from src.deepl import kno
from src.definition import T_NETWORK


class Networks:
    def unet(self, dim_lhs: int = 8, dim_rhs: int = 1) -> tuple[T_NETWORK, str]:
        return cno.UNet2d(in_channel=dim_lhs, out_channel=dim_rhs), "unet"

    def kno(self, dim_lhs: int = 8, dim_rhs: int = 1) -> tuple[T_NETWORK, str]:
        return kno.KNO2d(in_channel=dim_lhs, out_channel=dim_rhs), "kno"

    def fno(self, dim_lhs: int = 8, dim_rhs: int = 1) -> tuple[T_NETWORK, str]:
        return (
            fno.FNO(
                n_modes=(16, 16),
                hidden_channels=64,
                in_channels=dim_lhs,
                out_channels=dim_rhs,
            ),
            "fno",
        )

    def fno_ours(self, dim_lhs: int = 8, dim_rhs: int = 1) -> tuple[T_NETWORK, str]:
        return (
            fno_ours.FNO2d(n_channels_lhs=dim_lhs, n_channels_rhs=dim_rhs),
            "fno_ours",
        )

    def cno(self, dim_lhs: int = 8, dim_rhs: int = 1) -> tuple[T_NETWORK, str]:
        return cno.CNO2d(in_channel=dim_lhs, out_channel=dim_rhs), "cno"

    def networks(
        self, dim_lhs: int = 8, dim_rhs: int = 1
    ) -> typing.Generator[tuple[T_NETWORK, str], None, None]:
        yield self.fno(dim_lhs, dim_rhs)
        yield self.cno(dim_lhs, dim_rhs)
        yield self.kno(dim_lhs, dim_rhs)
        yield self.unet(dim_lhs, dim_rhs)
