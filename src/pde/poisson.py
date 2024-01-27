import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline

from src.numerics import grid
from src.util.saveload import SaveloadPde, SaveloadTorch

logger = logging.getLogger(__name__)


class PDEPoisson:
    def __init__(
        self, grid_x1: grid.Grid, grid_x2: grid.Grid, as_laplace: bool = False
    ):
        self._grid_x1, self._grid_x2 = grid_x1, grid_x2
        self._grids = grid.Grids([grid_x1, grid_x2])
        # TODO:
        #   check if we should divide this by 2
        self._sum_of_squares = (
            self._grid_x1.stepsize**2 + self._grid_x2.stepsize**2
        ) / 2

        self._source_term = self._make_source_term(as_laplace)

        self._lhss_bound: list[torch.Tensor] = []
        self._rhss_bound: list[float] = []
        self._lhss_internal: list[torch.Tensor] = []
        self._rhss_internal: list[float] = []
        self._sol = self._grids.zeroes_like()

        self._n_iters_max = int(5e3)
        self._error_threshold = 1e-4

    def _make_source_term(self, as_laplace: bool) -> torch.Tensor:
        if as_laplace:
            return np.zeros((self._grid_x1.n_pts, self._grid_x2.n_pts))

        res = self._grids.zeroes_like()
        for (
            (idx_x1, val_x1),
            (idx_x2, val_x2),
        ) in self._grids.steps_with_index():
            res[idx_x1, idx_x2] = np.sin(np.pi * val_x1) * np.sin(np.pi * val_x2)
        return res

    def solve(self) -> None:
        logger.info("Poisson.solve()")

        self._solve_boundary()
        self._solve_internal()
        self._register_internal()

    def _solve_boundary(self) -> None:
        for (
            (idx_x1, val_x1),
            (idx_x2, val_x2),
        ) in self._grids.boundaries_with_index():
            self._lhss_bound.append([val_x1, val_x2])
            rhs = -1.0
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
        saveload = SaveloadPde("poisson")
        if not (saveload.exists_boundary() and saveload.exists_internal()):
            self.solve()
        dataset_boundary = saveload.dataset_boundary(self._lhss_bound, self._rhss_bound)
        dataset_internal = saveload.dataset_internal(
            self._lhss_internal, self._rhss_internal
        )
        return dataset_boundary, dataset_internal

    def as_interpolator(self) -> RectBivariateSpline:
        def make_target() -> None:
            self.solve()
            return self._sol

        saveload = SaveloadTorch("poisson")
        location = saveload.rebase_location("raw")
        return RectBivariateSpline(
            self._grid_x1.step(),
            self._grid_x2.step(),
            saveload.load_or_make(location, make_target),
        )

    def plot_2d(self) -> None:
        plt.figure(figsize=(8, 6))
        plt.contourf(*self._grids.coords_as_mesh(), self._sol, cmap="viridis")
        plt.colorbar(label="u(x, y)")
        plt.title("Poisson 2D")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.savefig("poisson-2d")

    def plot_3d(self) -> None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            *self._grids.coords_as_mesh(),
            self._sol,
            cmap="viridis",
            edgecolor="k",
        )
        ax.set_title("Poisson 3D")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("u(X, Y)")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.savefig("poisson-3d")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )

    pde = PDEPoisson(
        grid.Grid(n_pts=50, stepsize=0.1, start=0.0),
        grid.Grid(n_pts=50, stepsize=0.1, start=0.0),
    )
    pde.as_dataset()
