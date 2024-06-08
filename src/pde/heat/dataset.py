import typing

import matplotlib.pyplot as plt
import torch

from src.numerics import grid
from src.pde import dataset
from src.util import dataset as dataset_util
from src.util import plot


class DatasetHeat(dataset.DatasetPDE2d):
    N_CHANNELS = 2

    def __init__(
        self,
        grids: grid.Grids,
        grid_time: grid.GridTime,
        sample_weight_min: float = -1.0,
        sample_weight_max: float = 1.0,
        n_samples_per_instance: int = 4,
        reweight_samples: bool = True,
    ):
        super().__init__(grids, name_problem="heat", name_dataset="sum_of_sine")
        self._grid_time = grid_time

        self._n_samples = n_samples_per_instance
        self._sample_weight_min, self._sample_weight_max = (
            sample_weight_min,
            sample_weight_max,
        )
        self._weights_samples: torch.Tensor
        self._reweight_samples = reweight_samples

        k_vector = torch.arange(1, self._n_samples + 1).float()
        self._k_pi = k_vector * torch.pi
        self._sinsin = torch.sin(
            self._k_pi * self._coords_x1.unsqueeze(-1)
        ) * torch.sin(self._k_pi * self._coords_x2.unsqueeze(-1))

    def _calc_weights_samples(self) -> None:
        self._weights_samples = torch.distributions.uniform.Uniform(
            self._sample_weight_min, self._sample_weight_max
        ).sample([self._n_samples])

    def solve_instance(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._reweight_samples:
            self._calc_weights_samples()
        return (
            self.solve_at_time(self._grid_time.start),
            self.solve_at_time(self._grid_time.end),
        )

    def solve_at_time(self, time: float) -> torch.Tensor:
        return (
            self._weights_samples * torch.exp(-2 * self._k_pi**2 * time) * self._sinsin
        ).sum(-1)

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

    def plot_snapshots(self) -> None:
        self._calc_weights_samples()

        fig, axs = plt.subplots(1, 3, figsize=(12, 5), dpi=200)
        times = [
            self._grid_time.start,
            (self._grid_time.start + self._grid_time.end) / 2,
            self._grid_time.end,
        ]

        for ax, time in zip(axs, times):
            snapshot = self.solve_at_time(time).detach().numpy()
            ax.matshow(snapshot, cmap="jet")
            ax.set_title(f"$t = {time}$")
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")

        fig.tight_layout()
        fig.savefig(self._base_dir / "plot_snapshots.png")
        plt.close(fig)

    @staticmethod
    def plot_animation_samples() -> None:
        torch.manual_seed(42)

        grids = grid.Grids(
            [
                grid.Grid.from_start_end(64, start=-1.0, end=1.0),
                grid.Grid.from_start_end(64, start=-1.0, end=1.0),
            ],
        )
        grid_time = grid.GridTime(n_pts=100, stepsize=1e-4)

        for n_instances in [1, 2, 4, 8, 16]:
            DatasetHeat(
                grids,
                grid_time,
                n_samples_per_instance=n_instances,
                reweight_samples=False,
            ).plot_animation()


class DatasetMaskedSingleHeat(dataset.DatasetMaskedSingle):
    def __init__(self, dataset_raw: dataset.DatasetPDE2d, mask: dataset_util.Masker):
        super().__init__(dataset_raw, mask, mask_index=1)


class DatasetMaskedDoubleHeat(dataset.DatasetMaskedDouble):
    def __init__(
        self,
        dataset_raw: dataset.DatasetPDE2d,
        mask_u_end: dataset_util.Masker,
        mask_u_start: typing.Optional[dataset_util.Masker] = None,
    ):
        mask_u_start = mask_u_start or mask_u_end
        super().__init__(dataset_raw, (mask_u_end, mask_u_start), (1, 0))


if __name__ == "__main__":
    DatasetHeat.plot_animation_samples()
