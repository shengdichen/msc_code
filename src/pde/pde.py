import logging
import os
from collections.abc import Iterable
from typing import Generator, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.definition import DEFINITION
from src.util.gif import MakerGif

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


class Distance:
    def __init__(
        self,
        ours: Union[torch.Tensor, float, int],
        theirs: Union[torch.Tensor, float, int] = 0.0,
    ):
        if not isinstance(ours, torch.Tensor):
            if not isinstance(ours, float):
                ours = float(ours)
            ours = torch.tensor(ours)
        self._ours = ours

        if isinstance(theirs, int):
            theirs = float(theirs)
        self._theirs = theirs

    def mse(self) -> torch.Tensor:
        return torch.mean((self._ours - self._theirs) ** 2)

    def norm_lp(self, p: int) -> torch.Tensor:
        if p % 2:
            diffs = torch.abs(self._ours - self._theirs)
        else:
            diffs = self._ours - self._theirs

        return torch.sum(diffs**p) ** (1 / p)


class Grid:
    def __init__(self, n_pts: int, stepsize: float, start: float = 0.0):
        if n_pts < 1:
            raise ValueError("grid must have at least one grid-point")
        if stepsize <= 0.0:
            raise ValueError("stepsize must be positive")
        self._n_pts, self._stepsize = n_pts, stepsize

        self._start = start
        self._end = start + n_pts * stepsize

    @property
    def n_pts(self) -> int:
        return self._n_pts

    @property
    def stepsize(self) -> float:
        return self._stepsize

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    def step(self) -> Generator[float, None, None]:
        curr = self._start
        while curr < self._end:
            yield curr
            curr += self._stepsize


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

    def init_solution_zeros(self) -> np.ndarray:
        # np's 0th axis := human's y
        # np's 1st axis := human's x
        return np.zeros((self._n_gridpts_y, self._n_gridpts_x))


class GridTime:
    def __init__(self, n_steps: int, stepsize: float):
        self._n_steps = n_steps
        self._stepsize = stepsize

        # (pre-)pad (just) enough zeros to make all timesteps uniform length
        # NOTE:
        #   add 1 to n-steps since we might want to track states before AND
        #   after time-stepping, which yields a total of (n-steps + 1) states
        #   in total
        self._formatter_timestep = f"{{:0{int(np.ceil(np.log10(self._n_steps+1)))}}}"

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def stepsize(self) -> float:
        return self._stepsize

    def timestep_formatted(self, timestep: int) -> str:
        return f"{self._formatter_timestep}".format(timestep)

    def step(self) -> Generator[int, None, None]:
        # NOTE:
        #   the zeroth timestep, containing in particular the
        #   initial-condition, should be handled separately by user
        for timestep in range(1, self._n_steps + 1):
            yield timestep


class PDEPoisson:
    def __init__(self):
        self._grid = GridTwoD(
            n_gridpts_x=50, n_gridpts_y=40, stepsize_x=0.1, step_size_y=0.15
        )

        self._source_f = self._apply_source_function()

        self._sol = self._grid.init_solution_zeros()
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


class PDEHeat:
    def __init__(self, alpha: float = 0.1):
        self._grid = GridTwoD(
            n_gridpts_x=50, n_gridpts_y=50, stepsize_x=0.1, step_size_y=0.1
        )
        self._sol = self._grid.init_solution_zeros()

        self._gridtime = GridTime(n_steps=100, stepsize=0.01)

        # initial-condition: heat-source at center
        self._sol[self._grid.n_gridpts_x // 2, self._grid.n_gridpts_y // 2] = 100.0

        self._alpha = alpha

        self._output_dir = DEFINITION.BIN_DIR / "pde/heat"
        os.makedirs(self._output_dir, exist_ok=True)

    def solve(self) -> None:
        self.plot_3d(0)  # initial-state

        for timestep in self._gridtime.step():
            sol_curr = self._sol.copy()
            for i in range(1, self._grid.n_gridpts_x - 1):
                for j in range(1, self._grid.n_gridpts_y - 1):
                    self._sol[i, j] = sol_curr[
                        i, j
                    ] + self._alpha * self._gridtime.stepsize * (
                        (sol_curr[i + 1, j] - 2 * sol_curr[i, j] + sol_curr[i - 1, j])
                        / self._grid.stepsize_x**2
                        + (sol_curr[i, j + 1] - 2 * sol_curr[i, j] + sol_curr[i, j - 1])
                        / self._grid.stepsize_y**2
                    )

            self.plot_3d(timestep)

    def plot_3d(self, timestep: int) -> None:
        timestep_formatted = self._gridtime.timestep_formatted(timestep)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            self._grid.coords_x, self._grid.coords_y, self._sol, cmap="viridis"
        )
        ax.set_title(
            "Heat [time-step " f"{timestep_formatted}/{self._gridtime.n_steps}" "]"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Temperature")
        plt.tight_layout()

        plt.savefig(self._output_dir / f"./frame_{timestep_formatted}")
        plt.close()

    def plot_gif(self) -> None:
        MakerGif(source_dir=self._output_dir).make(self._output_dir / "heat.gif")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )

    pde = PDEHeat()
    pde.solve()
    pde.plot_gif()
