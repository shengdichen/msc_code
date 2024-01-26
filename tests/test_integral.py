import math

import pytest
import torch

from src.util.integral import IntegralMontecarlo


class Integrand:
    @staticmethod
    def simple_2d(point: torch.Tensor) -> float:
        return 2 * point[0] + 3 * point[1] ** 2


class TestIntegralMontecarlo:
    def _range_nonstandard(self) -> list[tuple[float, float]]:
        return [(1, 3), (-1, 2)]

    def test_calc_volume(self):
        assert (
            IntegralMontecarlo()._volume
            == IntegralMontecarlo([[0, 1], [0, 1]])._volume
            == 1.0
        )

        assert (
            IntegralMontecarlo([[0, 2], [0, 3]])._volume
            == IntegralMontecarlo([[0, 3], [0, 2]])._volume
            == 6.0
        )

        assert IntegralMontecarlo([[-1, +2], [-5, +10], [-20, +30]])._volume == 2250.0

        for ranges in [[[1, 0]], [[0, 1], [1, 0]], [[0, 1], [0, 1], [1, 0]]]:
            with pytest.raises(ValueError):
                IntegralMontecarlo(ranges)

    def test_samples(self):
        assert torch.equal(
            IntegralMontecarlo(n_samples=10)._samples,
            torch.Tensor(
                [
                    [0.0000, 0.0000],
                    [0.5000, 0.5000],
                    [0.7500, 0.2500],
                    [0.2500, 0.7500],
                    [0.3750, 0.3750],
                    [0.8750, 0.8750],
                    [0.6250, 0.1250],
                    [0.1250, 0.6250],
                    [0.1875, 0.3125],
                    [0.6875, 0.8125],
                ]
            ),
        )

        assert torch.equal(
            IntegralMontecarlo(self._range_nonstandard(), n_samples=10)._samples,
            torch.Tensor(
                [
                    [1.0000, -1.0000],
                    [2.0000, 0.5000],
                    [2.5000, -0.2500],
                    [1.5000, 1.2500],
                    [1.7500, 0.1250],
                    [2.7500, 1.6250],
                    [2.2500, -0.6250],
                    [1.2500, 0.8750],
                    [1.3750, -0.0625],
                    [2.3750, 1.4375],
                ]
            ),
        )

    def test_integral(self):
        for n_samples, val in zip([100, 400, 900], [1.9810, 1.9963, 1.9980]):
            assert math.isclose(
                IntegralMontecarlo(n_samples=n_samples).integrate(Integrand.simple_2d),
                val,
                abs_tol=0.0001,
            )

        for n_samples, val in zip([100, 400, 900], [41.7092, 41.9504, 41.9695]):
            assert math.isclose(
                IntegralMontecarlo(
                    self._range_nonstandard(), n_samples=n_samples
                ).integrate(Integrand.simple_2d),
                val,
                abs_tol=0.0001,
            )
