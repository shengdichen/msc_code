from typing import Callable, Optional

import numpy as np
import torch


class IntegralMontecarlo:
    def __init__(
        self,
        ranges: Optional[list[tuple[float, float]]] = None,
        n_samples: Optional[int] = None,
    ):
        self._ranges = ranges or [(0, 1), (0, 1)]
        self._dim = len(self._ranges)
        self._volume = self._calc_volume()

        self._n_samples = n_samples or 30**self._dim
        self._samples = self._make_samples()

    def _calc_volume(self) -> float:
        volume = 1.0
        for rg in self._ranges:
            if rg[1] <= rg[0]:
                raise ValueError
            volume *= rg[1] - rg[0]
        return volume

    def _make_samples(self) -> torch.Tensor:
        engine = torch.quasirandom.SobolEngine(dimension=self._dim)
        ranges_np = np.array(self._ranges)
        mins, maxs = ranges_np[:, 0], ranges_np[:, 1]

        return engine.draw(self._n_samples) * (maxs - mins) + mins

    def integrate(self, integrand: Callable[[torch.Tensor], float]) -> float:
        integral = 0.0
        for pt in self._samples:
            integral += integrand(pt)

        return integral * self._volume / self._n_samples

    @staticmethod
    def integrand_toy(point: torch.Tensor) -> torch.Tensor:
        return 2 * point[0] + 3 * point[1] ** 2
