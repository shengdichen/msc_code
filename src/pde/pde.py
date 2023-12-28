import logging
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class PDEUtil:
    @staticmethod
    def boundary_space(sol: np.ndarray, val: float | Iterable = 0) -> np.ndarray:
        if isinstance(val, Iterable):
            v_bottom, v_top, v_left, v_right = val
        else:
            v_bottom, v_top, v_left, v_right = [val] * 4

        sol[:, 0] = v_bottom
        sol[:, -1] = v_top
        sol[0, :] = v_left
        sol[-1, :] = v_right
        return sol


class GridTwoD:
    def __init__(
        self, n_gridpts_x: int, n_gridpts_y: int, stepsize_x: float, step_size_y: float
    ):
        self._n_gridpts_x, self._n_gridpts_y = n_gridpts_x, n_gridpts_y
        self._stepsize_x, self._stepsize_y = stepsize_x, step_size_y

        self._coords_x, self._coords_y = np.meshgrid(
            np.linspace(
                0, self._stepsize_x * (self._n_gridpts_x - 1), self._n_gridpts_x
            ),
            np.linspace(
                0, self._stepsize_y * (self._n_gridpts_y - 1), self._n_gridpts_y
            ),
        )

    @property
    def n_gridpts_x(self) -> int:
        return self._n_gridpts_x

    @property
    def n_gridpts_y(self) -> int:
        return self._n_gridpts_y

    @property
    def stepsize_x(self) -> float:
        return self._stepsize_x

    @property
    def stepsize_y(self) -> float:
        return self._stepsize_y

    @property
    def coords_x(self) -> np.ndarray:
        return self._coords_x

    @property
    def coords_y(self) -> np.ndarray:
        return self._coords_y


class PDEPoisson:
    def __init__(self):
        self._grid = GridTwoD(
            n_gridpts_x=50, n_gridpts_y=40, stepsize_x=0.1, step_size_y=0.15
        )

        self._source_f = self._apply_source_function()

        # np's 0th axis := human's y
        # np's 1st axis := human's x
        self._sol = np.zeros((self._grid.n_gridpts_y, self._grid.n_gridpts_x))
        PDEUtil.boundary_space(self._sol, -1)

        self._n_iters_max = int(5e3)
        self._error_threshold = 1e-4

    def _apply_source_function(self) -> np.ndarray:
        return np.sin(np.pi * self._grid.coords_x) * np.sin(np.pi * self._grid.coords_y)

    def solve(self) -> None:
        logger.info("Poisson.solve()")

        # TODO:
        #   check if we should divide this by 2
        sum_of_squares = (self._grid.stepsize_x**2 + self._grid.stepsize_y**2) / 2

        # REF:
        #   https://ubcmath.github.io/MATH316/fd/laplace.html#exercises-for-laplace-s-equation
        for i in range(self._n_iters_max):
            sol_current = np.copy(self._sol)

            # exploit row-major ordering (x-loop within y-loop)
            for idx_y in range(1, self._grid.n_gridpts_y - 1):
                for idx_x in range(1, self._grid.n_gridpts_x - 1):
                    self._sol[idx_y, idx_x] = 0.25 * (
                        self._sol[idx_y + 1, idx_x]
                        + self._sol[idx_y - 1, idx_x]
                        + self._sol[idx_y, idx_x + 1]
                        + self._sol[idx_y, idx_x - 1]
                        - sum_of_squares * self._source_f[idx_y, idx_x]
                    )

            max_update = np.max(np.absolute(self._sol - sol_current))
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

    def plot_2d(self) -> None:
        self.solve()

        plt.figure(figsize=(8, 6))
        plt.contourf(
            self._grid.coords_x, self._grid.coords_y, self._sol, cmap="viridis"
        )
        plt.colorbar(label="u(x, y)")
        plt.title("Poisson 2D")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.savefig("poisson-2d")

    def plot_3d(self) -> None:
        self.solve()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            self._grid.coords_x,
            self._grid.coords_y,
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
    pde = PDEPoisson()
    pde.plot_3d()
