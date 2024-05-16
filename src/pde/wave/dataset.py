import typing

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.numerics import grid
from src.pde import dataset
from src.util import dataset as dataset_util
from src.util import plot


class DatasetWave(dataset.DatasetPDE2d):
    N_CHANNELS = 2

    def __init__(
        self,
        grids: grid.Grids,
        grid_time: grid.GridTime,
        constant_r: float = 0.85,
        constant_c: float = 0.1,
        sample_weight_min: float = -1.0,
        sample_weight_max: float = 1.0,
        n_samples_per_instance=4,
        reweight_samples: bool = True,
    ):
        super().__init__(grids, name_problem="wave", name_dataset="sum_of_sine")
        self._grid_time = grid_time

        self._n_samples = n_samples_per_instance
        self._sample_weight_min, self._sample_weight_max = (
            sample_weight_min,
            sample_weight_max,
        )
        self._weights_samples: torch.Tensor
        self._reweight_samples = reweight_samples

        self._constant_r, self._constant_c = constant_r, constant_c

        self._k1k2, self._sin_1, self._sin_2 = self._pre_calc()

    def _pre_calc(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_1, k_2 = np.meshgrid(
            np.arange(1, self._n_samples + 1),
            np.arange(1, self._n_samples + 1),
            indexing="ij",
        )
        k_1 = torch.from_numpy(k_1)
        k_2 = torch.from_numpy(k_2)

        return (
            k_1**2 + k_2**2,
            torch.sin(k_1 * torch.pi * self._coords_x1.unsqueeze(-1).unsqueeze(-1)),
            torch.sin(k_2 * torch.pi * self._coords_x2.unsqueeze(-1).unsqueeze(-1)),
        )

    def _calc_weights_samples(self) -> None:
        self._weights_samples = torch.distributions.uniform.Uniform(
            self._sample_weight_min, self._sample_weight_max
        ).sample([self._n_samples] * self._grids.n_dims)

    def solve_instance(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._reweight_samples:
            self._calc_weights_samples()
        return (
            self.solve_at_time(self._grid_time.start),
            self.solve_at_time(self._grid_time.end),
        )

    def solve_at_time(self, time: float) -> torch.Tensor:
        result = torch.sum(
            self._weights_samples
            * self._k1k2 ** (-self._constant_r)
            * torch.cos(self._constant_c * self._k1k2**0.5 * torch.pi * time)
            * self._sin_1
            * self._sin_2,
            dim=(-2, -1),
        )
        return result

    def plot_animation(self) -> None:
        self._calc_weights_samples()

        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

        p = plot.PlotAnimation2D(self._grid_time)
        p.plot(
            [
                self.solve_at_time(time).detach().numpy()
                for time in self._grid_time.step(with_start=True)
            ],
            fig,
            ax,
            save_as=self._base_dir / f"samples_{self._n_samples}.mp4",
        )

        plt.close(fig)

    @staticmethod
    def plot_animation_samples() -> None:
        torch.manual_seed(42)

        grids = grid.Grids(
            [
                grid.Grid.from_start_end(64, start=0.0, end=1.0),
                grid.Grid.from_start_end(64, start=0.0, end=1.0),
            ],
        )
        grid_time = grid.GridTime(n_pts=100, stepsize=0.1)

        for n_instances in [1, 2, 4, 8, 16]:
            DatasetWave(
                grids,
                grid_time,
                n_samples_per_instance=n_instances,
                reweight_samples=False,
            ).plot_animation()


class DatasetMaskedSingleWave(dataset.DatasetMaskedSingle):
    def __init__(self, dataset_raw: dataset.DatasetPDE2d, mask: dataset_util.Masker):
        super().__init__(dataset_raw, mask, mask_index=1)


class DatasetMaskedDoubleWave(dataset.DatasetMaskedDouble):
    def __init__(
        self,
        dataset_raw: dataset.DatasetPDE2d,
        mask_u_end: dataset_util.Masker,
        mask_u_start: typing.Optional[dataset_util.Masker] = None,
    ):
        mask_u_start = mask_u_start or mask_u_end
        super().__init__(dataset_raw, (mask_u_end, mask_u_start), (1, 0))


if __name__ == "__main__":
    DatasetWave.plot_animation_samples()
