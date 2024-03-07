import abc
import collections
import logging
import math
import pathlib
import typing

import numpy as np
import scipy
import torch
from scipy.interpolate import RectBivariateSpline

from src.numerics import grid
from src.util import plot
from src.util.dataset import Masker
from src.util.saveload import SaveloadImage, SaveloadTorch

logger = logging.getLogger(__name__)


class DatasetPoisson:
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


class DatasetConstructed(DatasetPoisson):
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

        coords_x1, coords_x2 = self._grids.coords_as_mesh()
        self._coords_x1, self._coords_x2 = (
            torch.from_numpy(coords_x1),
            torch.from_numpy(coords_x2),
        )
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


class DatasetConstructedSinCos(DatasetConstructed):
    def _generate_instance(self) -> tuple[torch.Tensor, torch.Tensor]:
        solutions, sources = self._grids.zeroes_like(), self._grids.zeroes_like()
        for i_sample in range(self._n_samples_per_instance):
            weight_sin, weight_cos = torch.distributions.Uniform(low=-1, high=1).sample(
                torch.Size([2])
            )
            factor = i_sample * torch.pi
            matrix_sin, matrix_cos = (
                torch.sin(factor * self._coords_x1)
                * torch.sin(factor * self._coords_x2),
                torch.cos(factor * self._coords_x1)
                * torch.cos(factor * self._coords_x2),
            )
            solution = weight_sin * matrix_sin + weight_cos * matrix_cos
            solutions += solution
            sources += -((self._n_samples_per_instance * torch.pi) ** 2) * solution

        normalizer = self._n_samples_per_instance**2
        solutions /= normalizer
        sources /= normalizer
        return solutions, sources


class DatasetConstructedSin(DatasetConstructed):
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        saveload: SaveloadTorch,
        name_dataset: str,
        n_instances: int = 10,
        n_samples_per_instance=4,
        constant_factor: float = 10.0,
    ):
        super().__init__(
            grid_x1,
            grid_x2,
            saveload=saveload,
            name_dataset=name_dataset,
            n_instances=n_instances,
            n_samples_per_instance=4,
        )

        self._constant_factor = constant_factor

    def _generate_instance(self) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.distributions.Uniform(low=-1, high=1).sample(
            [self._n_samples_per_instance] * self._grids.n_dims
        )

        sample_x1, sample_x2 = self._sample_coords()
        idx_sum = sample_x1**2 + sample_x2**2

        coords_x1 = self._coords_x1[..., None, None]
        coords_x2 = self._coords_x2[..., None, None]
        product = (
            weights
            * torch.sin(torch.pi * sample_x1 * coords_x1)
            * torch.sin(torch.pi * sample_x2 * coords_x2)
        )
        const_r = 0.85
        source = (
            self._constant_factor * torch.pi / self._n_samples_per_instance**2
        ) * torch.sum(
            (idx_sum**const_r) * product,
            dim=(-2, -1),
        )
        solution = (
            self._constant_factor / torch.pi / self._n_samples_per_instance**2
        ) * torch.sum(
            (idx_sum ** (const_r - 1)) * product,
            dim=(-2, -1),
        )
        return source, solution

    def _sample_coords(self) -> tuple[torch.Tensor, torch.Tensor]:
        sample_x1, sample_x2 = np.meshgrid(
            torch.arange(1, self._n_samples_per_instance + 1).float(),
            torch.arange(1, self._n_samples_per_instance + 1).float(),
        )
        return torch.from_numpy(sample_x1), torch.from_numpy(sample_x2)


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


class DatasetSolver(DatasetPoisson):
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
