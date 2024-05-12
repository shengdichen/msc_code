import abc
import logging
import pathlib
import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import multivariate_normal

from src.definition import T_DATASET
from src.numerics import grid
from src.pde.dataset import DatasetMaskedSingle, DatasetPDE2d
from src.util import dataset as dataset_util
from src.util import plot
from src.util.dataset import Masker

logger = logging.getLogger(__name__)


class DatasetPoisson2d(DatasetPDE2d):
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
    N_CHANNELS_LHS = 8
    N_CHANNELS_RHS = 1
    MASK_IDX = 0

    def __init__(self, dataset_raw: DatasetPDE2d, mask: Masker):
        super().__init__(dataset_raw, mask)


class DatasetSin(DatasetPoisson2d):
    def __init__(
        self,
        grids: grid.Grids,
        constant_r: float = 0.85,
        constant_multiplier: float = 1.0,
        sample_weight_min: float = -1.0,
        sample_weight_max: float = 1.0,
        n_samples_per_instance=4,
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
        n_samples_per_instance=4,
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


class DatasetPoissonMaskedSolution:
    N_CHANNELS_LHS = 8
    N_CHANNELS_RHS = 1

    def __init__(
        self,
        grids: grid.Grids,
        dataset_raw: DatasetPoisson2d,
        mask: dataset_util.Masker,
    ):
        self._grids = grids
        self._coords = self._grids.coords_as_mesh_torch()
        self._cos_coords = self._grids.cos_coords_as_mesh_torch()
        self._sin_coords = self._grids.sin_coords_as_mesh_torch()

        self._dataset_raw = dataset_raw
        self._mask = mask
        self._name = f"{self._dataset_raw.name_dataset}--sol_{self._mask.name}"

        self._normalizer: dataset_util.Normalizer
        self._dataset: T_DATASET

    @property
    def name(self) -> str:
        return self._name

    @property
    def dataset(self) -> T_DATASET:
        return self._dataset

    @classmethod
    def load_split(
        cls,
        grids: grid.Grids,
        dataset_raw: DatasetPoisson2d,
        masks_eval: typing.Iterable[dataset_util.Masker],
        masks_train: typing.Iterable[dataset_util.Masker],
        n_instances_eval: int,
        n_instances_train: int,
        base_dir: pathlib.Path = pathlib.Path("."),
    ) -> tuple[
        typing.Sequence["DatasetPoissonMaskedSolution"],
        typing.Sequence["DatasetPoissonMaskedSolution"],
    ]:
        eval_raw, train_raw = dataset_raw.load_split(
            n_instances_eval=n_instances_eval,
            n_instances_train=n_instances_train,
            base_dir=base_dir,
        )

        evals = []
        for mask in masks_eval:
            ds = cls(grids, dataset_raw, mask)
            ds.make(
                eval_raw,
                save_as=base_dir / f"{ds.name}--eval_{n_instances_eval}.pth",
            )
            evals.append(ds)

        trains = []
        for mask in masks_train:
            ds = cls(grids, dataset_raw, mask)
            ds.make(
                train_raw,
                save_as=base_dir / f"{ds.name}--train_{n_instances_train}.pth",
            )
            trains.append(ds)

        return evals, trains

    def make(
        self,
        dataset: T_DATASET,
        components_to_last: bool = False,
        save_as: typing.Optional[pathlib.Path] = None,
    ) -> T_DATASET:
        if save_as and save_as.exists():
            self._dataset = torch.load(save_as)
            return self._dataset

        lhss, rhss = [], []
        for solution, source in dataset:
            lhss.append(
                torch.stack(
                    [
                        *self._coords,
                        *self._cos_coords,
                        *self._sin_coords,
                        solution,
                        source,
                    ]
                )
            )
            rhss.append(solution.unsqueeze(0))
        dataset = torch.utils.data.TensorDataset(torch.stack(lhss), torch.stack(rhss))
        self._dataset = self._apply_mask(self._normalize(dataset), self._mask)
        if components_to_last:
            self._dataset = dataset_util.Reorderer().components_to_last(self._dataset)

        if save_as:
            torch.save(self._dataset, save_as)

        return self._dataset

    def remake(
        self,
        n_instances: int,
    ) -> typing.Callable[[], T_DATASET]:
        def f() -> T_DATASET:
            return self.make(self._dataset_raw.as_dataset(n_instances))

        return f

    def _normalize(self, dataset: T_DATASET) -> T_DATASET:
        self._normalizer = dataset_util.Normalizer.from_dataset(dataset)
        return self._normalizer.normalize_dataset(dataset)

    def _apply_mask(self, dataset: T_DATASET, mask: dataset_util.Masker) -> T_DATASET:
        lhss, rhss = dataset_util.DatasetPde.from_dataset(dataset).lhss_rhss
        for lhs in lhss:
            lhs[-2] = mask.mask(lhs[-2])
        return torch.utils.data.TensorDataset(lhss, rhss)

    def plot_instance(
        self, dataset: T_DATASET, n_instances: int = 1
    ) -> mpl.figure.Figure:
        fig, (axs_unmasked, axs_masked) = plt.subplots(
            2, n_instances, figsize=(10, 7.3), dpi=200, subplot_kw={"aspect": 1.0}
        )
        colormap = mpl.colormaps["viridis"]
        putil = plot.PlotUtil(self._grids)

        for i, (lhss, rhss) in enumerate(dataset):
            solution_unmasked, solution_masked = rhss[:, :, 0], lhss[:, :, 0]
            ax_unmasked, ax_masked = axs_unmasked[i], axs_masked[i]
            ax_unmasked.set_title(f"$u_{i+1}$")
            ax_masked.set_title(f"$u_{i+1}$ masked")
            putil.plot_2d(ax_unmasked, solution_unmasked, colormap=colormap)
            putil.plot_2d(ax_masked, solution_masked, colormap=colormap)
            if i == n_instances - 1:
                break
        return fig


class DatasetPoissonMaskedSolutionSource:
    N_CHANNELS_LHS = 8
    N_CHANNELS_RHS = 2

    def __init__(self, grids: grid.Grids):
        self._coords = grids.coords_as_mesh_torch()
        self._cos_coords = grids.cos_coords_as_mesh_torch()
        self._sin_coords = grids.sin_coords_as_mesh_torch()

        self._dataset = T_DATASET
        self._normalizer: dataset_util.Normalizer

    @staticmethod
    def as_name(
        dataset: DatasetPoisson2d,
        mask_solution: dataset_util.Masker,
        mask_source: dataset_util.Masker,
    ) -> str:
        return (
            f"{dataset.name_dataset}--"
            f"sol_{mask_solution.name()}--"
            f"source_{mask_source.name()}"
        )

    def make(
        self,
        dataset: T_DATASET,
        mask_solution: dataset_util.Masker,
        mask_source: dataset_util.Masker,
        components_to_last: bool = False,
        save_as: typing.Optional[pathlib.Path] = None,
    ) -> T_DATASET:
        if save_as and save_as.exists():
            return torch.load(save_as)

        lhss, rhss = [], []
        for solution, source in dataset:
            lhss.append(
                torch.stack(
                    [
                        *self._coords,
                        *self._cos_coords,
                        *self._sin_coords,
                        solution,
                        source,
                    ]
                )
            )
            rhss.append(torch.stack([solution, source]))
        dataset = torch.utils.data.TensorDataset(torch.stack(lhss), torch.stack(rhss))
        self._dataset = self._apply_mask(
            self._normalize(dataset), mask_solution, mask_source
        )
        if components_to_last:
            self._dataset = dataset_util.Reorderer().components_to_last(self._dataset)

        if save_as:
            torch.save(self._dataset, save_as)

        return self._dataset

    def _normalize(self, dataset: T_DATASET) -> T_DATASET:
        self._normalizer = dataset_util.Normalizer.from_dataset(dataset)
        return self._normalizer.normalize_dataset(dataset)

    def _apply_mask(
        self,
        dataset: T_DATASET,
        mask_solution: dataset_util.Masker,
        mask_source: dataset_util.Masker,
    ) -> T_DATASET:
        lhss, rhss = dataset_util.DatasetPde.from_dataset(dataset).lhss_rhss
        for lhs in lhss:
            lhs[-2] = mask_solution.mask(lhs[-2])
            lhs[-1] = mask_source.mask(lhs[-1])
        return torch.utils.data.TensorDataset(lhss, rhss)
