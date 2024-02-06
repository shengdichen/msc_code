import logging
import math
import typing

import numpy as np
import scipy
import torch
from scipy.interpolate import RectBivariateSpline

from src.deepl import network
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

    lp = LearnerPoisson()
    lp.plot()
