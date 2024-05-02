import torch
import numpy as np

from src.definition import T_DATASET
from src.numerics import grid
from src.pde.dataset import DatasetPDE2d


class DatasetWave(DatasetPDE2d):
    def __init__(
        self,
        grids: grid.Grids,
        constant_r: float = 0.85,
        constant_c: float = 0.1,
        t_start: float = 0.0,
        t_end: float = 0.01,
        sample_weight_min: float = -1.0,
        sample_weight_max: float = 1.0,
        n_samples_per_instance=4,
    ):
        super().__init__(grids)

        self._t_start, self._t_end = t_start, t_end
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

    def subdir_bin(self) -> str:
        return "wave"

    def as_name(self) -> str:
        return self._name

    def solve_instance(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.solve_at_time(self._t_start), self.solve_at_time(self._t_end)

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
