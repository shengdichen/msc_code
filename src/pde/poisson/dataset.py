import abc
import logging
import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import multivariate_normal

from src.numerics import grid
from src.pde.dataset import (DatasetMaskedDouble, DatasetMaskedSingle,
                             DatasetPDE2d)
from src.util import dataset as dataset_util
from src.util.dataset import Masker

logger = logging.getLogger(__name__)


class DatasetPoisson2d(DatasetPDE2d):
    N_CHANNELS = 2

    def __init__(self, grids: grid.Grids, name_dataset: str):
        super().__init__(grids, name_problem="poisson", name_dataset=name_dataset)

    @abc.abstractmethod
    def plot(self, set_title_upper: bool = True) -> mpl.figure.Figure:
        raise NotImplementedError

    def _plot(self, title_upper: typing.Optional[str]) -> mpl.figure.Figure:
        fig = plt.figure(figsize=(17, 9), dpi=200)
        if title_upper:
            fig.suptitle(f'Dataset "{title_upper}"')

        self._plot_solution_row(fig.add_subplot(2, 4, 1, aspect=1.0))
        self._plot_solution_row(fig.add_subplot(2, 4, 2, aspect=1.0))
        self._plot_solution_row(fig.add_subplot(2, 4, 3, aspect=1.0))
        self._plot_solution_row(fig.add_subplot(2, 4, 4, aspect=1.0))

        solution, source = self.solve_instance()

        title_u = "$u(\\cdot, \\cdot)\\quad$"
        ax_1_1 = fig.add_subplot(2, 4, 5, aspect=1.0)
        ax_1_1.set_title(f"solution {title_u}")
        self._putil.plot_2d(ax_1_1, solution)
        ax_1_2 = fig.add_subplot(2, 4, 6, projection="3d")
        ax_1_2.set_title(f"solution {title_u}")
        self._putil.plot_3d(ax_1_2, solution, label_z="")

        title_f = "$f(\\cdot, \\cdot)\\quad$"
        ax_1_3 = fig.add_subplot(2, 4, 7, aspect=1.0)
        ax_1_3.set_title(f"source {title_f}")
        self._putil.plot_2d(ax_1_3, source)
        ax_1_4 = fig.add_subplot(2, 4, 8, projection="3d")
        ax_1_4.set_title(f"source {title_f}")
        self._putil.plot_3d(ax_1_4, source, label_z="")

        return fig

    def _plot_solution_row(self, ax: mpl.axes.Axes) -> None:
        solution, __ = self.solve_instance()
        title_u = "$u(\\cdot, \\cdot)\\quad$"
        ax.set_title(f"instance {title_u}")
        self._putil.plot_2d(ax, solution)


class DatasetMaskedSinglePoisson(DatasetMaskedSingle):
    def __init__(self, dataset_raw: DatasetPDE2d, mask: Masker):
        super().__init__(dataset_raw, mask, mask_index=0)


class DatasetMaskedDoublePoisson(DatasetMaskedDouble):
    def __init__(
        self,
        dataset_raw: DatasetPDE2d,
        mask_solution: dataset_util.Masker,
        mask_source: typing.Optional[dataset_util.Masker] = None,
    ):
        mask_source = mask_source or mask_solution
        super().__init__(
            dataset_raw, (mask_solution, mask_source), masks_indexes=(0, 1)
        )


class DatasetSin(DatasetPoisson2d):
    def __init__(
        self,
        grids: grid.Grids,
        constant_r: float = 0.85,
        constant_multiplier: float = 1.0,
        sample_weight_min: float = -1.0,
        sample_weight_max: float = 1.0,
        n_samples_per_instance: int = 4,
    ):
        super().__init__(grids, name_dataset="sum_of_sine")

        self._constant_r, self._constant_multiplier = constant_r, constant_multiplier

        self._n_samples_per_instance = n_samples_per_instance
        self._weight_min, self._weight_max = sample_weight_min, sample_weight_max

    def plot(self, set_title_upper: bool = True) -> mpl.figure.Figure:
        return self._plot("Sum of Sine" if set_title_upper else "")

    def solve_instance(self) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.distributions.Uniform(
            low=self._weight_min, high=self._weight_max
        ).sample([self._n_samples_per_instance] * self._grids.n_dims)

        sample_x1, sample_x2 = self._sample_coords()
        idx_sum = sample_x1**2 + sample_x2**2

        coords_x1 = self._coords_x1[..., None, None]
        coords_x2 = self._coords_x2[..., None, None]
        product = (
            weights
            * torch.sin(torch.pi * sample_x1 * coords_x1)
            * torch.sin(torch.pi * sample_x2 * coords_x2)
        )
        source = (
            self._constant_multiplier * torch.pi / self._n_samples_per_instance**2
        ) * torch.sum(
            (idx_sum**self._constant_r) * product,
            dim=(-2, -1),
        )
        solution = (
            self._constant_multiplier / torch.pi / self._n_samples_per_instance**2
        ) * torch.sum(
            (idx_sum ** (self._constant_r - 1)) * product,
            dim=(-2, -1),
        )
        return solution, source

    def _sample_coords(self) -> tuple[torch.Tensor, torch.Tensor]:
        sample_x1, sample_x2 = np.meshgrid(
            torch.arange(1, self._n_samples_per_instance + 1).float(),
            torch.arange(1, self._n_samples_per_instance + 1).float(),
        )
        return torch.from_numpy(sample_x1), torch.from_numpy(sample_x2)


class DatasetGauss(DatasetPoisson2d):
    def __init__(
        self,
        grids: grid.Grids,
        constant_multiplier: float = 1.0,
        rng_np: np.random.Generator = np.random.default_rng(),
        sample_mu_with_sobol: bool = False,
        sample_sigma_same: bool = False,
        sample_sigma_min: float = 0.04,
        sample_sigma_max: float = 0.13,
        sample_weight_min: float = 0.3,
        sample_weight_max: float = 0.7,
        n_samples_per_instance: int = 4,
    ):
        super().__init__(grids, name_dataset="sum_of_gauss")
        self._coords_x1, self._coords_x2 = self._grids.coords_as_mesh()

        self._constant_multiplier = constant_multiplier
        self._rng_np = rng_np

        self._mu_with_sobol = sample_mu_with_sobol
        self._sigma_same = sample_sigma_same
        self._sigma_min, self._sigma_max = sample_sigma_min, sample_sigma_max

        self._weight_min, self._weight_max = sample_weight_min, sample_weight_max
        self._n_samples_per_instance = n_samples_per_instance

    def plot(self, set_title_upper: bool = True) -> mpl.figure.Figure:
        return self._plot("Sum of Gaussians" if set_title_upper else "")

    def solve_instance(self) -> tuple[torch.Tensor, torch.Tensor]:
        solution_final, source_final = (
            self._grids.zeroes_like_numpy(),
            self._grids.zeroes_like_numpy(),
        )
        for weight, mus, sigmas in zip(*self._make_sample_data()):
            solution, source = self._make_sample(mus, sigmas)
            solution_final += weight * solution
            source_final += weight * source
        return torch.from_numpy(solution_final), torch.from_numpy(source_final)

    def _make_sample_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = self._rng_np.uniform(
            low=self._weight_min,
            high=self._weight_max,
            size=self._n_samples_per_instance,
        )

        if self._mu_with_sobol:
            mu_vectors = self._grids.samples_sobol(self._n_samples_per_instance).numpy()
        else:
            mu_vectors = self._grids.sample_uniform(
                size=self._n_samples_per_instance, rng=self._rng_np
            )

        if self._sigma_same:
            sigmas = self._rng_np.uniform(
                low=self._sigma_min,
                high=self._sigma_max,
                size=self._n_samples_per_instance,
            )
            sigma_vectors = np.stack([sigmas] * self._grids.n_dims, axis=-1)
        else:
            sigma_vectors = self._rng_np.uniform(
                low=self._sigma_min,
                high=self._sigma_max,
                size=(self._n_samples_per_instance, self._grids.n_dims),
            )

        return weights, mu_vectors, sigma_vectors

    def _make_sample(
        self, mu_vec: np.ndarray, sigmas: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._constant_multiplier
            * self._calc_solution(mu_vec, sigma_mat=np.diag(sigmas**2)),
            self._constant_multiplier * self._calc_source(mu_vec, sigmas),
        )

    def _calc_solution(self, mu_vec: np.ndarray, sigma_mat: np.ndarray) -> np.ndarray:
        return multivariate_normal(mean=mu_vec, cov=sigma_mat).pdf(
            np.dstack((self._coords_x1, self._coords_x2))
        )

    def _calc_solution_ours(self, mu_vec: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        r_var_1, r_var_2 = 1 / (sigmas[0] ** 2), 1 / (sigmas[1] ** 2)
        diff_s_1, diff_s_2 = (
            (self._coords_x1 - mu_vec[0]) ** 2,
            (self._coords_x2 - mu_vec[1]) ** 2,
        )
        return (
            1
            / (2 * np.pi * np.prod(sigmas))
            * np.exp(-0.5 * (r_var_1 * diff_s_1 + r_var_2 * diff_s_2))
        )

    def _calc_source(self, mu_vec: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        r_var_1, r_var_2 = 1 / (sigmas[0] ** 2), 1 / (sigmas[1] ** 2)
        diff_s_1, diff_s_2 = (
            (self._coords_x1 - mu_vec[0]) ** 2,
            (self._coords_x2 - mu_vec[1]) ** 2,
        )
        return (
            1
            / (2 * np.pi * np.prod(sigmas))
            * (r_var_1**2 * diff_s_1 - r_var_1 + r_var_2**2 * diff_s_2 - r_var_2)
            * np.exp(-0.5 * (r_var_1 * diff_s_1 + r_var_2 * diff_s_2))
        )
