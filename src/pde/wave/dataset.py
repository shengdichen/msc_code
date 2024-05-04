import matplotlib.pyplot as plt
import numpy as np
import torch

from src.definition import DEFINITION, T_DATASET
from src.numerics import grid
from src.pde.dataset import DatasetPDE2d
from src.util import plot


class DatasetWave(DatasetPDE2d):
    def __init__(
        self,
        grids: grid.Grids,
        grid_time: grid.GridTime,
        constant_r: float = 0.85,
        constant_c: float = 0.1,
        sample_weight_min: float = -1.0,
        sample_weight_max: float = 1.0,
        n_samples_per_instance=4,
    ):
        super().__init__(grids, base_dir=DEFINITION.BIN_DIR / "wave")
        self._grid_time = grid_time

        self._name = "wave-sum_of_sine"

        self._n_samples = n_samples_per_instance
        self._weights_samples = torch.distributions.uniform.Uniform(
            sample_weight_min, sample_weight_max
        ).sample([n_samples_per_instance] * self._grids.n_dims)

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

    def as_name(self) -> str:
        return self._name

    def solve_instance(self) -> tuple[torch.Tensor, torch.Tensor]:
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

    def as_dataset(self, n_instances: int) -> T_DATASET:
        starts, ends = [], []

        for u_start, u_end in self.solve(n_instances):
            starts.append(u_start)
            ends.append(u_end)
        return torch.utils.data.TensorDataset(torch.stack(starts), torch.stack(ends))

    def plot_animation(self) -> None:
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
                grids, grid_time, n_samples_per_instance=n_instances
            ).plot_animation()


if __name__ == "__main__":
    DatasetWave.plot_animation_samples()
