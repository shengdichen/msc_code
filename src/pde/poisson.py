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

from src.deepl import cno, fno_2d, network
from src.definition import DEFINITION
from src.numerics import distance, grid, multidiff
from src.util import plot
from src.util.dataset import Masker, MaskerRandom
from src.util.saveload import SaveloadImage, SaveloadTorch

logger = logging.getLogger(__name__)


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

    def _make_dataset_unmasked(self) -> torch.utils.data.dataset.TensorDataset:
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


class LearnerPoissonFNO:
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        network_fno: torch.nn.Module,
        dataset_eval: torch.utils.data.dataset.TensorDataset,
        dataset_train: torch.utils.data.dataset.TensorDataset,
        saveload: SaveloadTorch,
        name_learner: str,
    ):
        self._device = DEFINITION.device_preferred

        self._grid_x1, self._grid_x2 = grid_x1, grid_x2
        self._grids = grid.Grids([self._grid_x1, self._grid_x2])
        self._network = network_fno.to(self._device)
        self._dataset_eval, self._dataset_train = dataset_eval, dataset_train

        self._saveload, self._name_learner = saveload, name_learner
        self._location = self._saveload.rebase_location(name_learner)

    def train(self, n_epochs: int = 2001, freq_eval: int = 100) -> None:
        optimizer = torch.optim.Adam(self._network.parameters(), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        for epoch in range(n_epochs):
            loss_all = []
            for lhss_batch, rhss_batch in torch.utils.data.DataLoader(
                self._dataset_train, batch_size=2
            ):
                lhss_batch, rhss_batch = (
                    lhss_batch.to(device=self._device, dtype=torch.float),
                    rhss_batch.to(device=self._device, dtype=torch.float),
                )
                optimizer.zero_grad()
                rhss_ours = self._network(lhss_batch)  # 10, 1001, 1
                loss_batch = distance.Distance(rhss_ours, rhss_batch).mse()
                loss_batch.backward()
                optimizer.step()

                loss_all.append(loss_batch.item())

            scheduler.step()
            if epoch % freq_eval == 0:
                print(f"train> mse: {np.average(loss_all)}")
                self.eval()

    def load_network_trained(
        self, n_epochs: int = 2001, freq_eval: int = 100, save_as_suffix: str = "model"
    ) -> torch.nn.Module:
        def make() -> torch.nn.Module:
            self.train(n_epochs=n_epochs, freq_eval=freq_eval)
            return self._network

        location = (
            self._saveload.rebase_location(f"{self._name_learner}--{save_as_suffix}")
            if save_as_suffix
            else self._location
        )
        self._network = self._saveload.load_or_make(location, make)
        return self._network

    def eval(self, print_result: bool = True) -> float:
        mse_abs_all, mse_rel_all = [], []
        with torch.no_grad():
            self._network.eval()
            for lhss_batch, rhss_batch in torch.utils.data.DataLoader(
                self._dataset_eval
            ):
                lhss_batch, rhss_batch = (
                    lhss_batch.to(device=self._device, dtype=torch.float),
                    rhss_batch.to(device=self._device, dtype=torch.float),
                )
                rhss_ours = self._network(lhss_batch)
                dst = distance.Distance(rhss_ours, rhss_batch)
                mse_abs_all.append(dst.mse().item())
                mse_rel_all.append(dst.mse_relative().item())
        mse_abs_avg, mse_rel_avg = np.average(mse_abs_all), np.average(mse_rel_all)
        if print_result:
            print(f"eval> (mse, mse%): {mse_abs_avg}, {mse_rel_avg}")

        return mse_rel_avg.item()

    @abc.abstractmethod
    def plot(self) -> None:
        pass

    def _plot_save(self, rhss_ours: torch.Tensor, save_as: str) -> None:
        plotter = plot.PlotFrame(
            self._grids, rhss_ours, save_as, SaveloadImage(self._saveload.base)
        )
        plotter.plot_2d()
        plotter.plot_3d()

    def _separate_lhss_rhss(
        self, dataset: torch.utils.data.dataset.TensorDataset
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lhss, rhss = [], []
        for lhs, rhs in dataset:
            lhss.append(lhs)
            rhss.append(rhs)
        return torch.stack(lhss), torch.stack(rhss)

    def _one_lhss_rhss(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
        index: int = 0,
        flatten_first_dimension: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lhss, rhss = list(dataset)[index]
        if not flatten_first_dimension:
            return lhss.unsqueeze(0), rhss.unsqueeze(0)
        return lhss, rhss


class LearnerPoissonFNO2d(LearnerPoissonFNO):
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        dataset_eval: torch.utils.data.dataset.TensorDataset,
        dataset_train: torch.utils.data.dataset.TensorDataset,
        saveload: SaveloadTorch,
        name_learner: str,
    ):
        super().__init__(
            grid_x1,
            grid_x2,
            fno_2d.FNO2d(n_channels_lhs=4),
            dataset_eval=dataset_eval,
            dataset_train=dataset_train,
            saveload=saveload,
            name_learner=f"network_fno_2d--{name_learner}",
        )

    def plot(self) -> None:
        lhss, rhss = self._one_lhss_rhss(self._dataset_eval)
        self._plot_save(rhss[0, :, :, 0], f"{self._location}-theirs")

        lhss = lhss.to(device=self._device, dtype=torch.float)
        rhss_ours = self._network(lhss).detach().to("cpu")[0, :, :, 0]
        self._plot_save(rhss_ours, f"{self._location}-ours")


class DatasetReorderCNO:
    def __init__(self, dataset: torch.utils.data.dataset.TensorDataset):
        self._dataset = dataset

    def reorder(self) -> torch.utils.data.dataset.TensorDataset:
        lhss, rhss = [], []
        for lhs, rhs in self._dataset:
            lhss.append(lhs.permute(2, 0, 1))
            rhss.append(rhs.permute(2, 0, 1))
        return torch.utils.data.TensorDataset(torch.stack(lhss), torch.stack(rhss))


class LearnerPoissonCNO2d(LearnerPoissonFNO):
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        dataset_eval: torch.utils.data.dataset.TensorDataset,
        dataset_train: torch.utils.data.dataset.TensorDataset,
        saveload: SaveloadTorch,
        name_learner: str,
    ):
        super().__init__(
            grid_x1,
            grid_x2,
            cno.CNO2d(in_channel=4, out_channel=1),
            dataset_eval=DatasetReorderCNO(dataset_eval).reorder(),
            dataset_train=DatasetReorderCNO(dataset_train).reorder(),
            saveload=saveload,
            name_learner=f"network_cno_2d--{name_learner}",
        )

    def plot(self) -> None:
        lhss, rhss = self._one_lhss_rhss(self._dataset_eval)
        self._plot_save(rhss[0, 0, :, :], f"{self._location}-theirs")

        lhss = lhss.to(device=self._device, dtype=torch.float)
        rhss_ours = self._network(lhss).detach().to("cpu")[0, 0, :, :]
        self._plot_save(rhss_ours, f"{self._location}-ours")


class LearnerPoissonFC:
    def __init__(self, n_pts_mask: int = 30):
        self._device = DEFINITION.device_preferred
        self._name_variant = f"fc-{n_pts_mask}"

        grid_x1 = grid.Grid(n_pts=50, stepsize=0.1, start=0.0)
        grid_x2 = grid.Grid(n_pts=50, stepsize=0.1, start=0.0)
        self._grids_full = grid.Grids([grid_x1, grid_x2])

        self._saveload = SaveloadTorch("poisson")
        solver = SolverPoisson(
            grid_x1, grid_x2, source=self._grids_full.constants_like(-200)
        )
        self._solver = solver.as_interpolator(
            saveload=self._saveload,
            name="dataset-fc",
            boundary_mean=-20,
            boundary_sigma=1,
        )
        self._lhss_eval, self._rhss_exact_eval = self._make_lhss_rhss_train(
            self._grids_full, n_pts=5000
        )

        grids_mask = grid.Grids(
            [
                grid.Grid(n_pts=n_pts_mask, stepsize=0.1, start=0.5),
                grid.Grid(n_pts=n_pts_mask, stepsize=0.1, start=0.5),
            ]
        )
        self._lhss_train, self._rhss_exact_train = self._make_lhss_rhss_train(
            grids_mask, n_pts=4000
        )

        self._network = network.Network(dim_x=2, with_time=False).to(self._device)
        self._eval_network = self._make_eval_network(use_multidiff=False)

    def _make_lhss_rhss_train(
        self, grids: grid.Grids, n_pts: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lhss = grids.samples_sobol(n_pts)
        rhss = torch.from_numpy(self._solver.ev(lhss[:, 0], lhss[:, 1])).view(-1, 1)
        return lhss.to(self._device), rhss.to(self._device)

    def _make_eval_network(
        self, use_multidiff: bool = True
    ) -> typing.Callable[[torch.Tensor], torch.Tensor]:
        def f(lhss: torch.Tensor) -> torch.Tensor:
            if use_multidiff:
                mdn = multidiff.MultidiffNetwork(self._network, lhss, ["x1", "x2"])
                return mdn.diff("x1", 2) + mdn.diff("x2", 2)
            return self._network(lhss)

        return f

    def train(self, n_epochs: int = 50001) -> None:
        optimiser = torch.optim.Adam(self._network.parameters())
        for epoch in range(n_epochs):
            optimiser.zero_grad()
            loss = distance.Distance(
                self._eval_network(self._lhss_train), self._rhss_exact_train
            ).mse()
            loss.backward()
            optimiser.step()

            if epoch % 100 == 0:
                logger.info(f"epoch {epoch:04}> " f"loss [train]: {loss.item():.4} ")
                self.evaluate_model()

        self._saveload.save(
            self._network,
            self._saveload.rebase_location(f"network-{self._name_variant}"),
        )

    def evaluate_model(self) -> None:
        dist = distance.Distance(
            self._eval_network(self._lhss_eval), self._rhss_exact_eval
        )
        logger.info(f"eval> (mse, mse%): {dist.mse()}, {dist.mse_percentage()}")

    def load(self) -> None:
        location = self._saveload.rebase_location(f"network-{self._name_variant}")
        self._network = self._saveload.load(location)

    def plot(self) -> None:
        res = self._grids_full.zeroes_like_numpy()

        for (
            (idx_x1, val_x1),
            (idx_x2, val_x2),
        ) in self._grids_full.steps_with_index():
            lhs = torch.tensor([val_x1, val_x2]).view(1, 2).to(self._device)
            res[idx_x1, idx_x2] = self._eval_network(lhs)

        plotter = plot.PlotFrame(
            self._grids_full,
            res,
            f"poisson-{self._name_variant}",
            SaveloadImage(self._saveload.base),
        )
        plotter.plot_2d()
        plotter.plot_3d()


class Learners:
    def __init__(self, n_instances_eval: int = 300, n_instances_train=100):
        self._grid_x1 = grid.Grid(n_pts=64, stepsize=0.01, start=0.0)
        self._grid_x2 = grid.Grid(n_pts=64, stepsize=0.01, start=0.0)

        self._n_instances_eval, self._n_instances_train = (
            n_instances_eval,
            n_instances_train,
        )

        self._saveload = SaveloadTorch("poisson")
        self._rng = np.random.default_rng(seed=42)

    def dataset_standard(self) -> None:
        name_dataset = "dataset-fno-2d-standard"
        ds = DatasetSolver(
            self._grid_x1,
            self._grid_x2,
            saveload=self._saveload,
            name_dataset=name_dataset,
            source=grid.Grids([self._grid_x1, self._grid_x2]).constants_like(-200),
            boundary_mean=-20,
            boundary_sigma=1,
        )

        learner = LearnerPoissonFNO2d(
            self._grid_x1,
            self._grid_x2,
            dataset_eval=ds.dataset_masked(mask_solution=MaskerRandom(0.5)),
            dataset_train=ds.dataset_masked(mask_solution=MaskerRandom(0.5)),
            saveload=self._saveload,
            name_learner="standard",
        )
        learner.train()
        learner.plot()

    def dataset_custom(
        self,
        n_samples_per_instance: int = 3,
    ) -> None:
        name, ds_size = "custom_sin", 1000
        indexes_eval, indexes_train = self._indexes_eval_train(ds_size)

        ds = DatasetConstructedSin(
            self._grid_x1,
            self._grid_x2,
            saveload=self._saveload,
            name_dataset=name,
            n_instances=ds_size,
            n_samples_per_instance=n_samples_per_instance,
        )
        ds_eval_raw = ds.dataset_raw_split(
            indexes=indexes_eval,
            save_as_suffix="eval",
        )
        ds_train_raw = ds.dataset_raw_split(
            indexes=indexes_train,
            save_as_suffix="train",
        )

        errors = []
        for perc_to_mask in np.arange(start=0.1, stop=1.0, step=0.1):
            ds_eval_masked = ds.dataset_masked(
                from_dataset=ds_eval_raw,
                mask_solution=MaskerRandom(perc_to_mask=perc_to_mask),
                save_as_suffix=f"eval_{self._n_instances_eval}",
            )
            ds_train_masked = ds.dataset_masked(
                from_dataset=ds_train_raw,
                mask_solution=MaskerRandom(perc_to_mask=perc_to_mask),
                save_as_suffix=f"train_{self._n_instances_train}",
            )

            learner = LearnerPoissonFNO2d(
                self._grid_x1,
                self._grid_x2,
                dataset_eval=ds_eval_masked,
                dataset_train=ds_train_masked,
                saveload=self._saveload,
                name_learner=name,
            )
            learner.load_network_trained(
                n_epochs=1001,
                save_as_suffix=f"random_{perc:.2}",
            )
            errors.append(learner.eval(print_result=False))

    def _indexes_eval_train(self, size_datset: int) -> tuple[np.ndarray, np.ndarray]:
        # NOTE:
        # generate indexes in one call with |replace| set to |False| to guarantee strict
        # separation of train and eval datasets
        indexes = self._rng.choice(
            size_datset,
            self._n_instances_eval + self._n_instances_train,
            replace=False,
        )
        return (
            indexes[: self._n_instances_eval],
            indexes[-self._n_instances_train :],
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )

    torch.manual_seed(42)

    learners = Learners()
    learners.dataset_custom()
