import abc
import collections
import logging
import math
import typing

import numpy as np
import scipy
import torch
from scipy.interpolate import RectBivariateSpline

from src.deepl import fno_2d, network
from src.definition import DEFINITION
from src.numerics import distance, grid, multidiff
from src.util import plot
from src.util.saveload import SaveloadPde, SaveloadTorch

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

        self._lhss_bound: list[torch.Tensor] = []
        self._rhss_bound: list[float] = []
        self._rhss_bound_in_mesh = self._grids.zeroes_like()
        self._lhss_internal: list[torch.Tensor] = []
        self._rhss_internal: list[float] = []
        self._sol = self._grids.zeroes_like()

        self._n_iters_max, self._error_threshold = n_iters_max, error_threshold

        self._saveload = SaveloadPde("poisson")

    def solve(self, boundary_mean: float = -1.0, boundary_sigma: float = 0.0) -> None:
        logger.info("Poisson.solve()")

        self._solve_boundary(mean=boundary_mean, sigma=boundary_sigma)
        self._solve_internal()
        self._register_internal()

    def _solve_boundary(self, mean: float, sigma: float) -> None:
        for (
            (idx_x1, val_x1),
            (idx_x2, val_x2),
        ) in self._grids.boundaries_with_index():
            self._lhss_bound.append([val_x1, val_x2])
            if math.isclose(sigma, 0.0):
                rhs = mean
            else:
                rhs = scipy.stats.norm.rvs(loc=mean, scale=sigma)
            self._rhss_bound.append(rhs)
            self._sol[idx_x1, idx_x2] = rhs
            self._rhss_bound_in_mesh[idx_x1, idx_x2] = rhs

    def _solve_internal(self) -> None:
        # REF:
        #   https://ubcmath.github.io/MATH316/fd/laplace.html#exercises-for-laplace-s-equation
        for i in range(self._n_iters_max):
            sol_current = np.copy(self._sol)
            self._solve_internal_current()

            max_update = torch.max(torch.abs(self._sol - sol_current))
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
        for (idx_x1, _), (idx_x2, _) in self._grids.internals_with_index():
            self._sol[idx_x1, idx_x2] = 0.25 * (
                self._sol[idx_x1 + 1, idx_x2]
                + self._sol[idx_x1 - 1, idx_x2]
                + self._sol[idx_x1, idx_x2 + 1]
                + self._sol[idx_x1, idx_x2 - 1]
                - self._sum_of_squares * self._source_term[idx_x1, idx_x2]
            )

    def _register_internal(self) -> None:
        for (
            (idx_x1, val_x1),
            (idx_x2, val_x2),
        ) in self._grids.internals_with_index():
            self._lhss_internal.append([val_x1, val_x2])
            self._rhss_internal.append(self._sol[idx_x1, idx_x2])

    @property
    def rhss_bound_in_mesh(self) -> torch.Tensor:
        return self._rhss_bound_in_mesh

    @property
    def rhss_in_mesh(self) -> torch.Tensor:
        return self._sol

    def as_interpolator(
        self, dataset_save_as: str = "dataset", **kwargs
    ) -> RectBivariateSpline:
        def make_target() -> None:
            self.solve(**kwargs)
            return self._sol

        location = self._saveload.rebase_location(dataset_save_as)
        self._sol = self._saveload.load_or_make(location, make_target)
        return RectBivariateSpline(
            self._grid_x1.step(),
            self._grid_x2.step(),
            self._sol,
        )

    def plot(self) -> None:
        plotter = plot.PlotFrame(self._grids, self._sol, "poisson")
        plotter.plot_2d()
        plotter.plot_3d()


class DatasetFNO:
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        source: torch.Tensor,
        boundary_mean: float,
        boundary_sigma: float,
        n_instances_train: int = 9,
        n_instances_test: int = 1,
    ):
        self._grid_x1, self._grid_x2 = grid_x1, grid_x2
        self._grids_full = grid.Grids([self._grid_x1, self._grid_x2])
        self._source = source
        self._boundary_mean, self._boundary_sigma = boundary_mean, boundary_sigma

        self._n_instances, self._n_instances_train, self._n_instances_test = (
            n_instances_train + n_instances_test,
            n_instances_train,
            n_instances_test,
        )
        self._saveload = SaveloadTorch("poisson")

    def dataset(self) -> torch.utils.data.dataset.TensorDataset:
        def make_target() -> torch.utils.data.dataset.TensorDataset:
            return self.make()

        location = self._saveload.rebase_location("dataset-fno-2d")
        return self._saveload.load_or_make(location, make_target)

    def make(self) -> torch.utils.data.dataset.TensorDataset:
        raise NotImplementedError

    def _generate_instances(self) -> collections.abc.Iterable[torch.Tensor]:
        raise NotImplementedError

    def _repeat_mesh_like(self, mesh_like: torch.Tensor, count: int) -> torch.Tensor:
        if mesh_like.dim() != 2:
            raise ValueError("expected mesh-like tensor of dimension 2")
        res = mesh_like.unsqueeze(dim=0)
        if count != 1:
            res = res.repeat(count, 1, 1)
        return res


class DatasetConstructed:
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        n_instances: int = 10,
        n_samples_per_instance=4,
    ):
        self._grid_x1, self._grid_x2 = grid_x1, grid_x2
        self._grids = grid.Grids([self._grid_x1, self._grid_x2])
        coords_x1, coords_x2 = self._grids.coords_as_mesh()
        self._coords_x1, self._coords_x2 = (
            torch.from_numpy(coords_x1),
            torch.from_numpy(coords_x2),
        )

        self._n_instancs = n_instances
        self._n_samples_per_instance = n_samples_per_instance  # aka, |K|

    def dataset(
        self, idx_min: int, idx_max: int
    ) -> torch.utils.data.dataset.TensorDataset:
        solutions, sources, solutions_masked = [], [], []
        for __ in range(self._n_instancs):
            solution, source = self._generate_one_instance()
            solutions.append(solution)
            sources.append(source)
            solutions_masked.append(self._grids.mask(solution, idx_min, idx_max))

        lhss = torch.stack(
            [
                torch.stack(solutions_masked),
                torch.stack(sources),
                self._coords_x1.repeat(self._n_instancs, 1, 1),
                self._coords_x2.repeat(self._n_instancs, 1, 1),
            ],
            dim=-1,
        )
        rhss = torch.stack(solutions).unsqueeze(-1)
        return torch.utils.data.TensorDataset(lhss, rhss)

    def _generate_one_instance(self) -> tuple[torch.Tensor, torch.Tensor]:
        solutions, sources = self._grids.zeroes_like(), self._grids.zeroes_like()
        for i_sample in range(self._n_samples_per_instance):
            weight_sin, weight_cos = torch.distributions.Uniform(low=-1, high=1).sample(
                [2]
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

    def plot_one(self) -> None:
        solution, __ = self._generate_one_instance()
        plotter = plot.PlotFrame(self._grids, solution, "poisson-dataset-custom")
        plotter.plot_2d()
        plotter.plot_3d()


class DatasetStandard(DatasetFNO):
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        idx_min: int,
        idx_max: int,
        source: torch.Tensor,
        boundary_mean: float,
        boundary_sigma: float,
        n_instances_train: int = 9,
        n_instances_test: int = 1,
    ):
        super().__init__(
            grid_x1,
            grid_x2,
            source,
            boundary_mean=boundary_mean,
            boundary_sigma=boundary_sigma,
            n_instances_train=n_instances_train,
            n_instances_test=n_instances_test,
        )

        self._idx_min, self._idx_max = idx_min, idx_max

    def make(self) -> torch.utils.data.dataset.TensorDataset:
        __, rhss_all, rhss_masked = self._generate_instances()
        coords_x1_all, coords_x2_all = [
            self._repeat_mesh_like(torch.from_numpy(coords_axis), self._n_instances)
            for coords_axis in self._grids_full.coords_as_mesh()
        ]
        source_all = self._repeat_mesh_like(self._source, self._n_instances)

        lhss_all = torch.stack(
            [source_all, rhss_masked, coords_x1_all, coords_x2_all], dim=-1
        )
        rhss_all = rhss_all.unsqueeze(dim=-1)
        return torch.utils.data.TensorDataset(lhss_all, rhss_all)

    def _generate_instances(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bounds_all, rhss_all, rhss_masked = [], [], []
        for __ in range(self._n_instances):
            solver = SolverPoisson(self._grid_x1, self._grid_x2, source=self._source)
            solver.solve(
                boundary_mean=self._boundary_mean, boundary_sigma=self._boundary_sigma
            )
            bounds_all.append(solver.rhss_bound_in_mesh)
            rhss_all.append(solver.rhss_in_mesh)
            rhss_masked.append(
                self._grids_full.mask(
                    solver.rhss_in_mesh, idx_min=self._idx_min, idx_max=self._idx_max
                )
            )

        return (
            torch.stack(bounds_all),
            torch.stack(rhss_all),
            torch.stack(rhss_masked),
        )


class LearnerPoissonFNO:
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        network_fno: torch.nn.Module,
        dataset_full: torch.utils.data.dataset.TensorDataset,
        dataset_mask: torch.utils.data.dataset.TensorDataset,
        saveload_location: str,
    ):
        self._device = DEFINITION.device_preferred

        self._grid_x1, self._grid_x2 = grid_x1, grid_x2
        self._grids = grid.Grids([self._grid_x1, self._grid_x2])
        self._network = network_fno.to(self._device)
        self._dataset_full, self._dataset_mask = dataset_full, dataset_mask

        self._saveload = SaveloadTorch("poisson")
        self._location = self._saveload.rebase_location(saveload_location)

    def train(self, n_epochs: int = 2001, freq_eval: int = 100) -> None:
        optimizer = torch.optim.Adam(self._network.parameters(), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        for epoch in range(n_epochs):
            loss_all = []
            for lhss_batch, rhss_batch in torch.utils.data.DataLoader(
                self._dataset_mask, batch_size=2
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

        self._saveload.save(self._network, self._location)

    def load(self) -> None:
        self._network = self._saveload.load(self._location)

    def eval(self) -> None:
        mse_abs_all, mse_rel_all = [], []
        with torch.no_grad():
            self._network.eval()
            for lhss_batch, rhss_batch in torch.utils.data.DataLoader(
                self._dataset_full
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
        print(f"eval> (mse, mse%): {mse_abs_avg}, {mse_rel_avg}")

    @abc.abstractmethod
    def plot(self) -> None:
        pass

    def _plot_save(self, rhss_ours: torch.Tensor, save_as: str) -> None:
        plotter = plot.PlotFrame(self._grids, rhss_ours, save_as)
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
        dataset_full: torch.utils.data.dataset.TensorDataset,
        dataset_mask: torch.utils.data.dataset.TensorDataset,
        name_dataset: str,
    ):
        super().__init__(
            grid_x1,
            grid_x2,
            fno_2d.FNO2d(n_channels_lhs=4),
            dataset_full=dataset_full,
            dataset_mask=dataset_mask,
            saveload_location=f"network-fno-2d-{name_dataset}",
        )
        self._name_dataset = name_dataset

    def plot(self) -> None:
        lhss, rhss = self._one_lhss_rhss(self._dataset_full)
        self._plot_save(rhss[0, :, :, 0], "poisson-fno-2d-theirs")

        lhss = lhss.to(device=self._device, dtype=torch.float)
        rhss_ours = self._network(lhss).detach().to("cpu")[0, :, :, 0]
        self._plot_save(rhss_ours, f"poisson-fno-2d-{self._name_dataset}")


class LearnerPoissonFC:
    def __init__(self, n_pts_mask: int = 30):
        self._device = DEFINITION.device_preferred
        self._name_variant = f"fc-{n_pts_mask}"

        grid_x1 = grid.Grid(n_pts=50, stepsize=0.1, start=0.0)
        grid_x2 = grid.Grid(n_pts=50, stepsize=0.1, start=0.0)
        self._grids_full = grid.Grids([grid_x1, grid_x2])

        solver = SolverPoisson(
            grid_x1, grid_x2, source=self._grids_full.constants_like(-200)
        )
        self._solver = solver.as_interpolator(
            dataset_save_as="dataset-fc", boundary_mean=-20, boundary_sigma=1
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

        saveload = SaveloadTorch("poisson")
        saveload.save(
            self._network, saveload.rebase_location(f"network-{self._name_variant}")
        )

    def evaluate_model(self) -> None:
        dist = distance.Distance(
            self._eval_network(self._lhss_eval), self._rhss_exact_eval
        )
        logger.info(f"eval> (mse, mse%): {dist.mse()}, {dist.mse_percentage()}")

    def load(self) -> None:
        saveload = SaveloadTorch("poisson")
        location = saveload.rebase_location(f"network-{self._name_variant}")
        self._network = saveload.load(location)

    def plot(self) -> None:
        res = self._grids_full.zeroes_like_numpy()

        for (
            (idx_x1, val_x1),
            (idx_x2, val_x2),
        ) in self._grids_full.steps_with_index():
            lhs = torch.tensor([val_x1, val_x2]).view(1, 2).to(self._device)
            res[idx_x1, idx_x2] = self._eval_network(lhs)

        plotter = plot.PlotFrame(self._grids_full, res, f"poisson-{self._name_variant}")
        plotter.plot_2d()
        plotter.plot_3d()


class Learners:
    def __init__(self):
        self._grid_x1 = grid.Grid(n_pts=50, stepsize=0.1, start=0.0)
        self._grid_x2 = grid.Grid(n_pts=50, stepsize=0.1, start=0.0)

        self._idx_min, self._idx_max = 10, 40

    def dataset_standard(self) -> None:
        ds = DatasetStandard(
            self._grid_x1,
            self._grid_x2,
            idx_min=self._idx_min,
            idx_max=self._idx_max,
            source=grid.Grids([self._grid_x1, self._grid_x2]).constants_like(-200),
            boundary_mean=-20,
            boundary_sigma=1,
        ).dataset()

        learner = LearnerPoissonFNO2d(
            self._grid_x1,
            self._grid_x2,
            dataset_full=ds,
            dataset_mask=ds,
            name_dataset="standard",
        )
        learner.train()
        learner.plot()

    def dataset_custom(self) -> None:
        ds = DatasetConstructed(self._grid_x1, self._grid_x2).dataset(
            self._idx_min, self._idx_max
        )

        learner = LearnerPoissonFNO2d(
            self._grid_x1,
            self._grid_x2,
            dataset_full=ds,
            dataset_mask=ds,
            name_dataset="custom",
        )
        learner.train()
        learner.plot()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )

    torch.manual_seed(42)

    learners = Learners()
    learners.dataset_custom()
