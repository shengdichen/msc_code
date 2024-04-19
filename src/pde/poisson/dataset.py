import abc
import collections
import logging
import math
import pathlib
import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from scipy.interpolate import RectBivariateSpline
from scipy.stats import multivariate_normal

from src.definition import T_DATASET
from src.numerics import grid
from src.pde.dataset import DatasetPDE2d
from src.util import dataset as dataset_util
from src.util import plot
from src.util.dataset import Masker
from src.util.saveload import SaveloadImage, SaveloadTorch

logger = logging.getLogger(__name__)


class DatasetPoisson2d(DatasetPDE2d):
    def solve(
        self, n_instances: int
    ) -> typing.Iterable[tuple[torch.Tensor, torch.Tensor]]:
        for __ in range(n_instances):
            yield self.solve_instance()

    def solve_instance(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        return (solution, source)
        """
        raise NotImplementedError

    def as_dataset(self, n_instances: int) -> torch.utils.data.dataset.TensorDataset:
        solutions, sources = [], []
        for solution, source in self.solve(n_instances):
            solutions.append(solution)
            sources.append(source)
        return torch.utils.data.TensorDataset(
            torch.stack(solutions), torch.stack(sources)
        )

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


class DatasetSin(DatasetPoisson2d):
    def __init__(
        self,
        grids: grid.Grids,
        n_samples_per_instance=4,
        sample_weight_min: float = -1.0,
        sample_weight_max: float = 1.0,
        constant_r: float = 0.85,
        constant_factor: float = 1.0,
    ):
        super().__init__(grids)

        self._coords_x1, self._coords_x2 = self._grids.coords_as_mesh_torch()

        self._n_samples_per_instance = n_samples_per_instance
        self._weight_min, self._weight_max = sample_weight_min, sample_weight_max
        self._constant_r, self._constant_factor = constant_r, constant_factor

    def as_name(self) -> str:
        return "sum_of_sine"

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
            self._constant_factor * torch.pi / self._n_samples_per_instance**2
        ) * torch.sum(
            (idx_sum**self._constant_r) * product,
            dim=(-2, -1),
        )
        solution = (
            self._constant_factor / torch.pi / self._n_samples_per_instance**2
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
        n_samples_per_instance=4,
        constant_factor: float = 1.0,
        rng_np: np.random.Generator = np.random.default_rng(42),
        sample_weight_min: float = 0.3,
        sample_weight_max: float = 0.7,
        sample_mu_with_sobol: bool = False,
        sample_sigma_same: bool = False,
        sample_sigma_min: float = 0.04,
        sample_sigma_max: float = 0.13,
    ):
        super().__init__(grids)

        self._constant_factor = constant_factor
        self._rng_np = rng_np
        self._n_samples_per_instance = n_samples_per_instance
        self._coords_x1, self._coords_x2 = self._grids.coords_as_mesh()

        self._weight_min, self._weight_max = sample_weight_min, sample_weight_max
        self._mu_with_sobol = sample_mu_with_sobol
        self._sigma_same = sample_sigma_same
        self._sigma_min, self._sigma_max = sample_sigma_min, sample_sigma_max

    def as_name(self) -> str:
        return "sum_of_gauss"

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
        self, mu_vec: np.ndarray, sigmas: torch.utils.data.dataset.TensorDataset
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._constant_factor
            * self._calc_solution(mu_vec, sigma_mat=np.diag(sigmas**2)),
            self._constant_factor * self._calc_source(mu_vec, sigmas),
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

    def __init__(self, grids: grid.Grids):
        self._coords = grids.coords_as_mesh_torch()
        self._cos_coords = grids.cos_coords_as_mesh_torch()
        self._sin_coords = grids.sin_coords_as_mesh_torch()

        self._normalizer: dataset_util.Normalizer
        self._dataset: T_DATASET
        self._putil = plot.PlotUtil(grids)

    @staticmethod
    def _name(dataset: DatasetPoisson2d, mask: dataset_util.Masker) -> str:
        return f"{dataset.as_name()}--sol_{mask.as_name()}"

    def load_split(
        self,
        dataset: DatasetPoisson2d,
        masks_eval: typing.Iterable[dataset_util.Masker],
        masks_train: typing.Iterable[dataset_util.Masker],
        n_instances_eval: int,
        n_instances_train: int,
        base_dir: pathlib.Path = pathlib.Path("."),
    ):
        eval_raw, train_raw = dataset.load_split(
            n_instances_eval=n_instances_eval,
            n_instances_train=n_instances_train,
            base_dir=base_dir,
        )

        evals = []
        for mask in masks_eval:
            path = (
                base_dir / f"{self._name(dataset, mask)}--eval_{n_instances_eval}.pth"
            )
            evals.append(self.make(eval_raw, mask, save_as=path))

        trains = []
        for mask in masks_train:
            path = (
                base_dir / f"{self._name(dataset, mask)}--train_{n_instances_train}.pth"
            )
            trains.append(self.make(train_raw, mask, save_as=path))

        return evals, trains

    def make(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
        mask: dataset_util.Masker,
        components_to_last: bool = False,
        save_as: typing.Optional[pathlib.Path] = None,
    ) -> torch.utils.data.dataset.TensorDataset:
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
            rhss.append(solution.unsqueeze(0))
        dataset = torch.utils.data.TensorDataset(torch.stack(lhss), torch.stack(rhss))
        self._dataset = self._apply_mask(self._normalize(dataset), mask)
        if components_to_last:
            self._dataset = dataset_util.Reorderer().components_to_last(self._dataset)

        if save_as:
            torch.save(self._dataset, save_as)

        return self._dataset

    def _normalize(
        self, dataset: torch.utils.data.dataset.TensorDataset
    ) -> torch.utils.data.dataset.TensorDataset:
        self._normalizer = dataset_util.Normalizer.from_dataset(dataset)
        return self._normalizer.normalize_dataset(dataset)

    def _apply_mask(
        self, dataset: torch.utils.data.dataset.TensorDataset, mask: dataset_util.Masker
    ) -> torch.utils.data.dataset.TensorDataset:
        lhss, rhss = dataset_util.DatasetPde.from_dataset(dataset).lhss_rhss
        for lhs in lhss:
            lhs[-2] = mask.mask(lhs[-2])
        return torch.utils.data.TensorDataset(lhss, rhss)

    def plot_instance(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
        n_instances: int = 1,
    ) -> mpl.figure.Figure:
        fig, (axs_unmasked, axs_masked) = plt.subplots(
            2, n_instances, figsize=(10, 7.3), dpi=200, subplot_kw={"aspect": 1.0}
        )
        colormap = mpl.colormaps["viridis"]

        for i, (lhss, rhss) in enumerate(dataset):
            solution_unmasked, solution_masked = rhss[:, :, 0], lhss[:, :, 0]
            ax_unmasked, ax_masked = axs_unmasked[i], axs_masked[i]
            ax_unmasked.set_title(f"$u_{i+1}$")
            ax_masked.set_title(f"$u_{i+1}$ masked")
            self._putil.plot_2d(ax_unmasked, solution_unmasked, colormap=colormap)
            self._putil.plot_2d(ax_masked, solution_masked, colormap=colormap)
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
    def _name(
        dataset: DatasetPoisson2d,
        mask_solution: dataset_util.Masker,
        mask_source: dataset_util.Masker,
    ) -> str:
        return (
            f"{dataset.as_name()}--"
            f"sol_{mask_solution.as_name()}--"
            f"source_{mask_source.as_name()}"
        )

    def make(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
        mask_solution: dataset_util.Masker,
        mask_source: dataset_util.Masker,
        components_to_last: bool = False,
        save_as: typing.Optional[pathlib.Path] = None,
    ) -> torch.utils.data.dataset.TensorDataset:
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

    def _normalize(
        self, dataset: torch.utils.data.dataset.TensorDataset
    ) -> torch.utils.data.dataset.TensorDataset:
        self._normalizer = dataset_util.Normalizer.from_dataset(dataset)
        return self._normalizer.normalize_dataset(dataset)

    def _apply_mask(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
        mask_solution: dataset_util.Masker,
        mask_source: dataset_util.Masker,
    ) -> torch.utils.data.dataset.TensorDataset:
        lhss, rhss = dataset_util.DatasetPde.from_dataset(dataset).lhss_rhss
        for lhs in lhss:
            lhs[-2] = mask_solution.mask(lhs[-2])
            lhs[-1] = mask_source.mask(lhs[-1])
        return torch.utils.data.TensorDataset(lhss, rhss)


class DatasetPoissonLegacy:
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        n_instances: int,
        name_dataset: str,
        saveload: SaveloadTorch,
    ):
        self._grid_x1, self._grid_x2 = grid_x1, grid_x2
        self._grids = grid.Grids([self._grid_x1, self._grid_x2])

        self._n_instances = n_instances
        self._saveload_location, self._saveload = (
            f"dataset--{name_dataset}",
            saveload,
        )

    @property
    def n_instances(self) -> int:
        return self._n_instances

    def dataset_raw(
        self,
    ) -> torch.utils.data.dataset.TensorDataset:
        def make() -> torch.utils.data.dataset.TensorDataset:
            return self._make_dataset_raw()

        return self._load_or_make(
            make,
            location=f"{self._saveload_location}--raw_{self._n_instances}",
        )

    @abc.abstractmethod
    def _make_dataset_raw(self) -> torch.utils.data.dataset.TensorDataset:
        raise NotImplementedError

    def dataset_raw_split(
        self,
        indexes: typing.Union[np.ndarray, collections.abc.Sequence[int]],
        autosave: typing.Optional[bool] = True,
        save_as_suffix: typing.Optional[typing.Union[str, pathlib.Path]] = None,
    ) -> torch.utils.data.dataset.TensorDataset:
        def make() -> torch.utils.data.dataset.TensorDataset:
            return torch.utils.data.Subset(self.dataset_raw(), indexes)

        return self._load_or_make(
            make,
            location=f"{self._saveload_location}--{save_as_suffix}_{len(indexes)}",
            autosave=autosave,
        )

    @abc.abstractmethod
    def dataset_masked(self) -> torch.utils.data.dataset.TensorDataset:
        raise NotImplementedError

    def _load_or_make(
        self,
        make: typing.Callable[..., torch.utils.data.dataset.TensorDataset],
        location: typing.Optional[typing.Union[str, pathlib.Path]] = None,
        autosave: typing.Optional[bool] = True,
    ) -> torch.utils.data.dataset.TensorDataset:
        if autosave:
            location = location or self._saveload_location
            return self._saveload.load_or_make(
                self._saveload.rebase_location(location), make
            )
        return make()

    def plot_instance(self) -> None:
        plotter = plot.PlotFrame(
            self._grids,
            self._generate_instance_solution(),
            self._saveload_location,
            SaveloadImage(self._saveload.base),
        )
        plotter.plot_2d()
        plotter.plot_3d()

    @abc.abstractmethod
    def _generate_instance_solution(self) -> torch.Tensor:
        raise NotImplementedError


class DatasetConstructed(DatasetPoissonLegacy):
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        saveload: SaveloadTorch,
        name_dataset: str,
        n_instances: int = 10,
        n_samples_per_instance=4,
    ):
        super().__init__(
            grid_x1,
            grid_x2,
            n_instances=n_instances,
            saveload=saveload,
            name_dataset=name_dataset,
        )

        self._coords_x1, self._coords_x2 = self._grids.coords_as_mesh_torch()
        self._n_samples_per_instance = n_samples_per_instance  # aka, |K|

    def _make_dataset_raw(self) -> torch.utils.data.dataset.TensorDataset:
        solutions, sources = [], []
        for __ in range(self._n_instances):
            solution, source = self._generate_instance()
            solutions.append(solution)
            sources.append(source)
        return torch.utils.data.TensorDataset(
            torch.stack(solutions), torch.stack(sources)
        )

    def dataset_masked(
        self,
        mask_source: typing.Optional[Masker] = None,
        mask_solution: typing.Optional[Masker] = None,
        from_dataset: typing.Optional[torch.utils.data.dataset.TensorDataset] = None,
        autosave: typing.Optional[bool] = False,
        save_as_suffix: typing.Optional[typing.Union[str, pathlib.Path]] = "masked",
    ) -> torch.utils.data.dataset.TensorDataset:
        def make() -> torch.utils.data.dataset.TensorDataset:
            dataset_raw = from_dataset or self.dataset_raw()
            n_instances = len(dataset_raw)

            solutions, sources, solutions_masked = [], [], []
            for solution, source in dataset_raw:
                solutions.append(solution)
                sources.append(mask_source.mask(source) if mask_source else source)
                solutions_masked.append(
                    mask_solution.mask(solution) if mask_solution else solution
                )
            return self._assemble(
                solutions_masked, sources, solutions, n_instances=n_instances
            )

        return self._load_or_make(
            make,
            location=f"{self._saveload_location}--{save_as_suffix}",
            autosave=autosave,
        )

    def _assemble(
        self,
        solutions_masked: torch.Tensor,
        sources: torch.Tensor,
        solutions: torch.Tensor,
        n_instances: typing.Optional[int] = None,
    ) -> torch.utils.data.dataset.TensorDataset:
        n_instances = n_instances or self._n_instances
        lhss = torch.stack(
            [
                torch.stack(solutions_masked),
                torch.stack(sources),
                self._coords_x1.repeat(n_instances, 1, 1),
                self._coords_x2.repeat(n_instances, 1, 1),
            ],
            dim=-1,
        )
        rhss = torch.stack(solutions).unsqueeze(-1)
        return torch.utils.data.TensorDataset(lhss, rhss)

    def _generate_instance_solution(self) -> torch.Tensor:
        return self._generate_instance()[0]

    @abc.abstractmethod
    def _generate_instance(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class SolverPoisson:
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        source: torch.Tensor,
        n_iters_max: int = int(5e3),
        error_threshold: float = 1e-4,
    ):
        self._grid_x1, self._grid_x2 = grid_x1, grid_x2
        self._grids = grid.Grids([grid_x1, grid_x2])
        # TODO:
        #   check if we should divide this by 2
        self._sum_of_squares = (
            self._grid_x1.stepsize**2 + self._grid_x2.stepsize**2
        ) / 2

        self._source_term = source
        self._solution = self._grids.zeroes_like()

        self._n_iters_max, self._error_threshold = n_iters_max, error_threshold

    def solve(
        self, boundary_mean: float = -1.0, boundary_sigma: float = 0.0
    ) -> torch.Tensor:
        logger.info("Poisson.solve()")
        self._solve_boundary(mean=boundary_mean, sigma=boundary_sigma)
        self._solve_internal()

        return self._solution

    def _solve_boundary(self, mean: float, sigma: float) -> None:
        for (idx_x1, __), (idx_x2, __) in self._grids.boundaries_with_index():
            if math.isclose(sigma, 0.0):
                rhs = mean
            else:
                rhs = scipy.stats.norm.rvs(loc=mean, scale=sigma)
            self._solution[idx_x1, idx_x2] = rhs

    def _solve_internal(self) -> None:
        # REF:
        #   https://ubcmath.github.io/MATH316/fd/laplace.html#exercises-for-laplace-s-equation
        for i in range(self._n_iters_max):
            sol_current = np.copy(self._solution)
            self._solve_internal_current()

            max_update = torch.max(torch.abs(self._solution - sol_current))
            if max_update < self._error_threshold:
                logger.info(
                    f"normal termination at iteration [{i}/{self._n_iters_max}]"
                    "\n\t"
                    f"update at termination [{max_update}]"
                )
                break
        else:
            logger.warning(
                f"forced termination at max-iteration limit [{self._n_iters_max}]"
                "\n\t"
                f"update at termination [{max_update}]"
            )

    def _solve_internal_current(self) -> None:
        for (idx_x1, __), (idx_x2, __) in self._grids.internals_with_index():
            self._solution[idx_x1, idx_x2] = 0.25 * (
                self._solution[idx_x1 + 1, idx_x2]
                + self._solution[idx_x1 - 1, idx_x2]
                + self._solution[idx_x1, idx_x2 + 1]
                + self._solution[idx_x1, idx_x2 - 1]
                - self._sum_of_squares * self._source_term[idx_x1, idx_x2]
            )

    def solution_internal(self) -> tuple[torch.Tensor, torch.Tensor]:
        lhss: list[tuple[float, float]] = []
        rhss: list[float] = []

        for (
            (idx_x1, val_x1),
            (idx_x2, val_x2),
        ) in self._grids.internals_with_index():
            lhss.append((val_x1, val_x2))
            rhss.append(self._solution[idx_x1, idx_x2].item())
        return torch.tensor(lhss), torch.tensor(rhss)

    def solution_boundary(
        self,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        lhss: list[tuple[float, float]] = []
        rhss_flat: list[float] = []
        rhss_mesh = self._grids.zeroes_like()

        for (
            (idx_x1, val_x1),
            (idx_x2, val_x2),
        ) in self._grids.boundaries_with_index():
            lhss.append((val_x1, val_x2))
            rhs = self._solution[idx_x1, idx_x2].item()
            rhss_flat.append(rhs)
            rhss_mesh[idx_x1, idx_x2] = rhs
        return torch.tensor(lhss), (torch.tensor(rhss_mesh), rhss_mesh)

    def as_interpolator(
        self, saveload: SaveloadTorch, name: str = "dataset", **kwargs
    ) -> RectBivariateSpline:
        def make_target() -> None:
            self.solve(**kwargs)
            return self._solution

        location = saveload.rebase_location(name)
        self._solution = saveload.load_or_make(location, make_target)
        return RectBivariateSpline(
            self._grid_x1.step(),
            self._grid_x2.step(),
            self._solution,
        )

    def plot(self, saveload: SaveloadImage, name: str = "poisson-solver") -> None:
        plotter = plot.PlotFrame(self._grids, self._solution, name, saveload)
        plotter.plot_2d()
        plotter.plot_3d()


class DatasetSolver(DatasetPoissonLegacy):
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        saveload: SaveloadTorch,
        name_dataset: str,
        source: torch.Tensor,
        boundary_mean: float,
        boundary_sigma: float,
        n_instances_train: int = 9,
        n_instances_test: int = 1,
    ):
        super().__init__(
            grid_x1,
            grid_x2,
            n_instances=n_instances_train + n_instances_test,
            saveload=saveload,
            name_dataset=name_dataset,
        )

        self._source = source
        self._boundary_mean, self._boundary_sigma = boundary_mean, boundary_sigma

        self._n_instances_train, self._n_instances_test = (
            n_instances_train,
            n_instances_test,
        )

    def _make_dataset_raw(self) -> torch.utils.data.dataset.TensorDataset:
        rhss_all, rhss_masked = self._generate_instances()
        self._assemble(rhss_all, rhss_masked)

    def dataset_masked(
        self,
        mask_solution: typing.Optional[Masker] = None,
        location: typing.Optional[typing.Union[str, pathlib.Path]] = None,
    ) -> torch.utils.data.dataset.TensorDataset:
        def make() -> None:
            rhss_all, rhss_masked = self._generate_instances(mask_solution)
            return self._assemble(rhss_all, rhss_masked)

        return self._load_or_make(make, location)

    def _generate_instances(
        self,
        mask_solution: typing.Optional[Masker] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rhss_all, rhss_masked = [], []
        for __ in range(self._n_instances):
            solver = SolverPoisson(
                self._grid_x1, self._grid_x2, source=self._source, error_threshold=1
            )
            solution = solver.solve(
                boundary_mean=self._boundary_mean, boundary_sigma=self._boundary_sigma
            )
            rhss_all.append(solution)
            rhss_masked.append(
                mask_solution.mask(solution) if mask_solution else solution
            )

        return torch.stack(rhss_all), torch.stack(rhss_masked)

    def _assemble(
        self, rhss_all: torch.Tensor, rhss_masked: torch.Tensor
    ) -> torch.utils.data.dataset.TensorDataset:
        source_all = self._repeat_mesh_like(self._source, self._n_instances)
        coords_x1_all, coords_x2_all = [
            self._repeat_mesh_like(torch.from_numpy(coords_axis), self._n_instances)
            for coords_axis in self._grids.coords_as_mesh()
        ]

        lhss_all = torch.stack(
            [source_all, rhss_masked, coords_x1_all, coords_x2_all], dim=-1
        )
        rhss_all = rhss_all.unsqueeze(dim=-1)
        return torch.utils.data.TensorDataset(lhss_all, rhss_all)

    def _generate_instance_solution(self) -> torch.Tensor:
        solver = SolverPoisson(self._grid_x1, self._grid_x2, source=self._source)
        return solver.solve(
            boundary_mean=self._boundary_mean, boundary_sigma=self._boundary_sigma
        )

    def _repeat_mesh_like(self, mesh_like: torch.Tensor, count: int) -> torch.Tensor:
        if mesh_like.dim() != 2:
            raise ValueError("expected mesh-like tensor of dimension 2")
        res = mesh_like.unsqueeze(dim=0)
        if count != 1:
            res = res.repeat(count, 1, 1)
        return res
