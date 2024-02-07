import itertools
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


class PDEPoisson:
    def __init__(self, grid_x1: grid.Grid, grid_x2: grid.Grid, source: torch.Tensor):
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

        self._n_iters_max = int(5e3)
        self._error_threshold = 1e-4

        self._saveload = SaveloadPde("poisson")

    def _make_source_term(self, as_laplace: bool) -> torch.Tensor:
        if as_laplace:
            return 100 * np.ones((self._grid_x1.n_pts, self._grid_x2.n_pts))

        res = self._grids.zeroes_like()
        for (
            (idx_x1, val_x1),
            (idx_x2, val_x2),
        ) in self._grids.steps_with_index():
            res[idx_x1, idx_x2] = np.sin(np.pi * val_x1) * np.sin(np.pi * val_x2)
        return res

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

    def as_dataset(
        self,
    ) -> tuple[
        torch.utils.data.dataset.TensorDataset, torch.utils.data.dataset.TensorDataset
    ]:
        if not (self._saveload.exists_boundary() and self._saveload.exists_internal()):
            self.solve()
        dataset_boundary = self._saveload.dataset_boundary(
            self._lhss_bound, self._rhss_bound
        )
        dataset_internal = self._saveload.dataset_internal(
            self._lhss_internal, self._rhss_internal
        )
        return dataset_boundary, dataset_internal

    def as_interpolator(self) -> RectBivariateSpline:
        def make_target() -> None:
            self.solve()
            return self._sol

        location = self._saveload.rebase_location("raw")
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
        n_instances_train: int = 9,
        n_instances_test: int = 1,
    ):
        self._grid_x1, self._grid_x2 = grid_x1, grid_x2
        self._grids_full = grid.Grids([self._grid_x1, self._grid_x2])
        self._source = source

        self._n_instances, self._n_instances_train, self._n_instances_test = (
            n_instances_train + n_instances_test,
            n_instances_train,
            n_instances_test,
        )
        self._saveload = SaveloadTorch("poisson")

    def as_dataset(self) -> torch.utils.data.dataset.TensorDataset:
        def make_target() -> torch.utils.data.dataset.TensorDataset:
            return self.make()

        location = self._saveload.rebase_location("dataset-fno")
        return self._saveload.load_or_make(location, make_target)

    def make(self) -> torch.utils.data.dataset.TensorDataset:
        raise NotImplementedError

    def _generate_instances(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _repeat_mesh_like(self, mesh_like: torch.Tensor, count: int) -> torch.Tensor:
        if mesh_like.dim() != 2:
            raise ValueError("expected mesh-like tensor of dimension 2")
        res = mesh_like.unsqueeze(dim=0)
        if count != 1:
            res = res.repeat(count, 1, 1)
        return res


class DatasetFNOMesh(DatasetFNO):
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        source: torch.Tensor,
        n_instances_train: int = 9,
        n_instances_test: int = 1,
    ):
        super().__init__(
            grid_x1,
            grid_x2,
            source,
            n_instances_train=n_instances_train,
            n_instances_test=n_instances_test,
        )

        self._lhss_x1, self._lhss_x2 = self._grids_full.coords_as_mesh()

    def make(self) -> torch.utils.data.dataset.TensorDataset:
        # coords_x1_all: (n_instances, n_gripts_x1, n_gripts_x2)
        # coords_x2_all: (n_instances, n_gripts_x1, n_gripts_x2)
        # rhss_all: (n_instances, n_gripts_x1, n_gripts_x2)
        bounds_all, rhss_all = self._generate_instances()
        coords_x1_all, coords_x2_all = [
            self._repeat_mesh_like(torch.from_numpy(coords_axis), self._n_instances)
            for coords_axis in self._grids_full.coords_as_mesh()
        ]
        source_all = self._repeat_mesh_like(self._source, self._n_instances)

        # make fno-ready:
        # lhss_all: (n_instances, n_gripts_x1, n_gripts_x2, 4)
        # rhss_all: (n_instances, n_gripts_x1, n_gripts_x2, 1)
        lhss_all = torch.stack(
            [bounds_all, source_all, coords_x1_all, coords_x2_all], dim=-1
        )
        rhss_all = rhss_all.unsqueeze(dim=-1)
        return torch.utils.data.TensorDataset(lhss_all, rhss_all)

    def _generate_instances(self) -> tuple:
        bounds_all, rhss_all = [], []
        for __ in range(self._n_instances):
            solver = PDEPoisson(self._grid_x1, self._grid_x2, source=self._source)
            solver.solve(boundary_mean=20, boundary_sigma=1)
            bounds_all.append(solver.rhss_bound_in_mesh)
            rhss_all.append(solver.rhss_in_mesh)

        return (
            torch.stack(bounds_all),
            torch.stack(rhss_all),
        )


class DatasetFNOInterpolation(DatasetFNO):
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        source: torch.Tensor,
        n_gridpts_per_instance: int = 5000,
        n_instances_train: int = 9,
        n_instances_test: int = 1,
    ):
        super().__init__(
            grid_x1,
            grid_x2,
            source,
            n_instances_train=n_instances_train,
            n_instances_test=n_instances_test,
        )

        self._lhss = self._grids_full.samples_sobol(n_gridpts_per_instance)

    def make(self) -> torch.utils.data.dataset.TensorDataset:
        # lhss_all: (n_instances, n_gripts, 2)
        # rhss_all: (n_instances, n_gripts, 1)
        lhss_all, rhss_all = self._generate_instances()

        # make fno-ready:
        # lhss_all: (n_instances, n_gripts, 3)
        lhss_all = torch.cat([rhss_all, lhss_all], dim=-1)

        return torch.utils.data.TensorDataset(lhss_all, rhss_all)

    def _generate_instances(self) -> tuple[torch.Tensor, torch.Tensor]:
        lhss_all, rhss_all = [], []
        for __ in range(self._n_instances_test + self._n_instances_train):
            lhss_all.append(self._lhss)
            rhss_all.append(self._generate_rhss_instance())

        lhss_all_torch, rhss_all_torch = torch.stack(lhss_all), torch.stack(rhss_all)
        return lhss_all_torch, rhss_all_torch

    def _generate_rhss_instance(self) -> torch.Tensor:
        solver = PDEPoisson(self._grid_x1, self._grid_x2, source=self._source)
        solver.solve(boundary_mean=20, boundary_sigma=1)
        rhss = torch.from_numpy(
            solver.as_interpolator().ev(self._lhss[:, 0], self._lhss[:, 1])
        ).view(-1, 1)
        return rhss


class MaskingDataset:
    def __init__(self, dataset: torch.utils.data.dataset.TensorDataset):
        self._dataset = dataset

        self._n_batches = len(self._dataset)
        self._shape_lhs, self._shape_rhs = (
            self._dataset[0][0].shape,
            self._dataset[0][1].shape,
        )
        self._n_gridpts_x1, self._n_gridpts_x2 = self._shape_lhs[0], self._shape_lhs[1]

    def mask(
        self, val_min: float, val_max: float, val_mask: float = 0.0
    ) -> torch.utils.data.dataset.TensorDataset:
        lhss_all, rhss_all = self._init_lhss_rhss_masked(val_mask)

        for i, (lhss, rhss) in enumerate(self._dataset):
            for idx_x1, idx_x2 in itertools.product(
                range(self._n_gridpts_x1), range(self._n_gridpts_x2)
            ):
                lhs, rhs = lhss[idx_x1, idx_x2, :], rhss[idx_x1, idx_x2, :]
                val_x1, val_x2 = lhs[2], lhs[3]
                if val_min < val_x1 < val_max and val_min < val_x2 < val_max:
                    lhss_all[i, idx_x1, idx_x2, :] = lhs
                    rhss_all[i, idx_x1, idx_x2, :] = rhs

        return torch.utils.data.TensorDataset(lhss_all, rhss_all)

    def _init_lhss_rhss_masked(
        self, val_mask: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if val_mask == 0.0:
            lhss_all, rhss_all = (
                torch.zeros((self._n_batches, *self._shape_lhs)),
                torch.zeros((self._n_batches, *self._shape_rhs)),
            )
        else:
            lhss_all, rhss_all = (
                val_mask * torch.ones((self._n_batches, *self._shape_lhs)),
                val_mask * torch.ones((self._n_batches, *self._shape_rhs)),
            )
        return lhss_all, rhss_all


class LearnerPoissonFNO:
    def __init__(self):
        self._device = DEFINITION.device_preferred

        self._grid_x1 = grid.Grid(n_pts=50, stepsize=0.1, start=0.0)
        self._grid_x2 = grid.Grid(n_pts=50, stepsize=0.1, start=0.0)

        ds_fno = DatasetFNOMesh(
            self._grid_x1,
            self._grid_x2,
            source=grid.Grids([self._grid_x1, self._grid_x2]).constants_like(50),
        )
        self._dataset_full = ds_fno.as_dataset()
        self._dataset_mask = MaskingDataset(self._dataset_full).mask(
            val_min=0.0, val_max=5.0
        )

        self._network = fno_2d.FNO2d(n_channels_lhs=4).to(self._device)

        self._saveload = SaveloadTorch("poisson")
        self._location = self._saveload.rebase_location("network-fno")

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
        mse_rel_all = []
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
                mse_rel_all.append(
                    distance.Distance(rhss_ours, rhss_batch).mse_relative().item()
                )
        mse_rel_avg = np.average(mse_rel_all)
        print(f"eval> mse%: {mse_rel_avg}")

    def plot(self) -> None:
        for lhss_batch, rhss_batch in torch.utils.data.DataLoader(self._dataset_full):
            lhss_batch, rhss_batch = (
                lhss_batch.to(device=self._device, dtype=torch.float),
                rhss_batch.to(device=self._device, dtype=torch.float),
            )
            rhss_ours = self._network(lhss_batch)[0, :, :, 0].detach().to("cpu")
            break

        plotter = plot.PlotFrame(
            grid.Grids([self._grid_x1, self._grid_x2]), rhss_ours, "poisson-fno"
        )
        plotter.plot_2d()
        plotter.plot_3d()


class LearnerPoisson:
    def __init__(self):
        self._device = DEFINITION.device_preferred

        grid_x1 = grid.Grid(n_pts=50, stepsize=0.1, start=0.0)
        grid_x2 = grid.Grid(n_pts=50, stepsize=0.1, start=0.0)
        self._grids_full = grid.Grids([grid_x1, grid_x2])

        self._solver = PDEPoisson(
            grid_x1, grid_x2, source=self._grids_full.constants_like(100)
        ).as_interpolator()

        self._lhss_eval = self._grids_full.samples_sobol(5000)
        self._rhss_exact_eval = torch.from_numpy(
            self._solver.ev(self._lhss_eval[:, 0], self._lhss_eval[:, 1])
        ).view(-1, 1)
        self._lhss_eval, self._rhss_exact_eval = (
            self._lhss_eval.to(self._device),
            self._rhss_exact_eval.to(self._device),
        )

        self._grids_train = grid.Grids(
            [
                grid.Grid(n_pts=40, stepsize=0.1, start=0.0),
                grid.Grid(n_pts=40, stepsize=0.1, start=0.0),
            ]
        )
        self._lhss_train = self._grids_train.samples_sobol(4000)
        self._rhss_exact_train = torch.from_numpy(
            self._solver.ev(self._lhss_train[:, 0], self._lhss_train[:, 1])
        ).view(-1, 1)
        self._lhss_train, self._rhss_exact_train = (
            self._lhss_train.to(self._device),
            self._rhss_exact_train.to(self._device),
        )

        self._network = network.Network(dim_x=2, with_time=False).to(self._device)
        self._optimiser = torch.optim.Adam(self._network.parameters())
        self._eval_network = self._make_eval_network(use_multidiff=False)

    def _make_eval_network(
        self, use_multidiff: bool = True
    ) -> typing.Callable[[torch.Tensor], torch.Tensor]:
        def f(lhss: torch.Tensor) -> torch.Tensor:
            if use_multidiff:
                mdn = multidiff.MultidiffNetwork(self._network, lhss, ["x1", "x2"])
                return mdn.diff("x1", 2) + mdn.diff("x2", 2)
            return self._network(lhss)

        return f

    def train(self, n_epochs: int = 10001) -> None:
        for epoch in range(n_epochs):
            loss = self._train_epoch()
            if epoch % 100 == 0:
                logger.info(f"epoch {epoch:04}> " f"loss [train]: {loss:.4} ")
                self.evaluate_model()

        saveload = SaveloadTorch("poisson")
        saveload.save(self._network, saveload.rebase_location("network"))

    def _train_epoch(self) -> float:
        self._optimiser.zero_grad()

        loss = distance.Distance(
            self._eval_network(self._lhss_train), self._rhss_exact_train
        ).mse()
        loss.backward()
        self._optimiser.step()

        return loss.item()

    def evaluate_model(self) -> None:
        dist = distance.Distance(
            self._eval_network(self._lhss_eval), self._rhss_exact_eval
        )
        logger.info(f"eval> (mse, mse%): {dist.mse()}, {dist.mse_percentage()}")

    def plot(self) -> None:
        saveload = SaveloadTorch("poisson")
        self._network = saveload.load(saveload.rebase_location("network"))

        res = self._grids_full.zeroes_like_numpy()

        for (
            (idx_x1, val_x1),
            (idx_x2, val_x2),
        ) in self._grids_full.steps_with_index():
            lhs = torch.tensor([val_x1, val_x2]).view(1, 2).to(self._device)
            res[idx_x1, idx_x2] = self._eval_network(lhs)

        plotter = plot.PlotFrame(self._grids_full, res, "poisson-ours")
        plotter.plot_2d()
        plotter.plot_3d()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )

    torch.manual_seed(42)

    lp_fno = LearnerPoissonFNO()
    lp_fno.load()
    lp_fno.plot()

    lp = LearnerPoisson()
    lp.plot()
