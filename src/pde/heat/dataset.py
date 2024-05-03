import pathlib

import matplotlib.pyplot as plt
import torch

from src.definition import DEFINITION, T_DATASET
from src.numerics import grid
from src.pde.dataset import DatasetPDE2d
from src.util import plot


class DatasetHeat(DatasetPDE2d):
    def __init__(
        self,
        grids: grid.Grids,
        t_start: float = 0.0,
        t_end: float = 0.01,
        sample_weight_min: float = -1.0,
        sample_weight_max: float = 1.0,
        n_samples_per_instance=4,
    ):
        super().__init__(grids)

        self._t_start, self._t_end = t_start, t_end
        self._name = "heat-sum_of_sine"

        self._n_samples = n_samples_per_instance
        self._weights_samples = torch.distributions.uniform.Uniform(
            sample_weight_min, sample_weight_max
        ).sample([self._n_samples])

        k_vector = torch.arange(1, self._n_samples + 1).float()
        self._k_pi = k_vector * torch.pi
        self._sinsin = torch.sin(
            self._k_pi * self._coords_x1.unsqueeze(-1)
        ) * torch.sin(self._k_pi * self._coords_x2.unsqueeze(-1))

    def subdir_bin(self) -> pathlib.Path:
        p = DEFINITION.BIN_DIR / "heat"
        p.mkdir(exist_ok=True)
        return p

    def as_name(self) -> str:
        return self._name

    def solve_instance(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.solve_at_time(self._t_start), self.solve_at_time(self._t_end)

    def solve_at_time(self, time: float) -> torch.Tensor:
        return (
            self._weights_samples * torch.exp(-2 * self._k_pi**2 * time) * self._sinsin
        ).sum(-1)

    def as_dataset(self, n_instances: int) -> T_DATASET:
        starts, ends = [], []

        for u_start, u_end in self.solve(n_instances):
            starts.append(u_start)
            ends.append(u_end)
        return torch.utils.data.TensorDataset(torch.stack(starts), torch.stack(ends))

    def plot_animation(self, grid_time: grid.GridTime) -> None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

        p = plot.PlotAnimation2D(grid_time)
        p.plot(
            [
                self.solve_at_time(time).detach().numpy()
                for time in grid_time.step(with_start=True)
            ],
            fig,
            ax,
            save_as=self.subdir_bin() / f"samples_{self._n_samples}.mp4",
        )

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
        grid_time = grid.GridTime(n_pts=100, stepsize=0.1)

        for n_instances in [1, 2, 4, 8, 16]:
            DatasetHeat(grids, n_samples_per_instance=n_instances).plot_animation(
                grid_time
            )


if __name__ == "__main__":
    DatasetHeat.plot_animation_samples()
