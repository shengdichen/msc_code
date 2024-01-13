import logging
import math
import os
import pathlib
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
        self._end = start + (n_pts - 1) * stepsize
        self._pts = [self._start + i * self._stepsize for i in range(self._n_pts)]
        self._boundaries = [self._start, self._end]

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

    def step(self, with_start: bool = True, with_end: bool = True) -> list[float]:
        start = 0 if with_start else 1
        if not with_end:
            return self._pts[start:-1]
        return self._pts[start:]

    def step_with_index(
        self, with_start: bool = True, with_end: bool = True
    ) -> Iterable[tuple[int, float]]:
        start = 0 if with_start else 1
        return enumerate(self.step(with_start, with_end), start=start)

    def is_on_boundary(self, val: float) -> bool:
        for boundary in self._boundaries:
            if math.isclose(val, boundary):
                return True
        return False

    def index_of(self, value: float) -> int:
        return self._pts.index(value)


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


class GridTime(Grid):
    def __init__(self, n_pts: int, stepsize: float, start=0.0):
        # NOTE:
        #   the zeroth timestep, containing in particular the
        #   initial-condition, should be handled separately by user
        # NOTE:
        #   add 1 to n-steps since we might want to track states before AND
        #   after time-stepping, which yields a total of (n-steps + 1) states
        #   in total
        super().__init__(n_pts=n_pts + 1, stepsize=stepsize, start=start)

        # (pre-)pad (just) enough zeros to make all timesteps uniform length
        self._formatter_timestep = f"{{:0{int(np.ceil(np.log10(self._n_pts)))}}}"

    @property
    def n_pts(self) -> int:
        return self._n_pts - 1

    def timestep_formatted(self, timestep: int) -> str:
        return f"{self._formatter_timestep}".format(timestep)

    def step(self, with_start: bool = False, with_end: bool = True) -> list[float]:
        return super().step(with_start, with_end)

    def step_with_index(
        self, with_start: bool = False, with_end: bool = True
    ) -> Iterable[tuple[int, float]]:
        return super().step_with_index(with_start, with_end)


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
    def __init__(
        self, grid_time: GridTime, grid_x1: Grid, grid_x2: Grid, alpha: float = 0.01
    ):
        self._grid_time, self._grid_x1, self._grid_x2 = grid_time, grid_x1, grid_x2

        self._lhss: list[torch.Tensor] = []
        self._rhss: list[float] = []
        self._snapshot = torch.zeros((self._grid_x1.n_pts, self._grid_x2.n_pts))

        self._alpha = alpha

        self._output_dir = DEFINITION.BIN_DIR / "pde/heat"
        os.makedirs(self._output_dir, exist_ok=True)

    def solve(self) -> None:
        for lhs, rhs in self._solve_init():
            self._lhss.append(lhs)
            self._rhss.append(rhs)
        self._plot_snapshot(0)

        for idx_t, val_t in self._grid_time.step_with_index():
            for lhs, rhs in self._solve_internal(val_t):
                self._lhss.append(lhs)
                self._rhss.append(rhs)
            for lhs, rhs in self._solve_bound(val_t):
                self._lhss.append(lhs)
                self._rhss.append(rhs)
            self._plot_snapshot(idx_t)

        self.save_raw()

    def _solve_init(self) -> Generator[tuple[torch.Tensor, float], None, None]:
        for idx_x1, val_x1 in self._grid_x1.step_with_index():
            for idx_x2, val_x2 in self._grid_x2.step_with_index():
                lhs = torch.tensor([self._grid_time.start, val_x1, val_x2])
                if (
                    idx_x1 == self._grid_x1.n_pts / 2
                    and idx_x2 == self._grid_x2.n_pts / 2
                ):
                    # initial-condition: heat-source at center
                    rhs = 100.0
                else:
                    rhs = 0

                self._snapshot[idx_x1, idx_x2] = rhs
                yield lhs, rhs

    def _solve_internal(
        self, val_t: float
    ) -> Generator[tuple[torch.Tensor, float], None, None]:
        sol_curr = self._snapshot.clone()

        for idx_x1, val_x1 in self._grid_x1.step_with_index(
            with_start=False, with_end=False
        ):
            for idx_x2, val_x2 in self._grid_x2.step_with_index(
                with_start=False, with_end=False
            ):
                lhs = torch.tensor([val_t, val_x1, val_x2])
                rhs = sol_curr[
                    idx_x1, idx_x2
                ] + self._alpha * self._grid_time.stepsize * (
                    (
                        sol_curr[idx_x1 + 1, idx_x2]
                        - 2 * sol_curr[idx_x1, idx_x2]
                        + sol_curr[idx_x1 - 1, idx_x2]
                    )
                    / self._grid_x1.stepsize**2
                    + (
                        sol_curr[idx_x1, idx_x2 + 1]
                        - 2 * sol_curr[idx_x1, idx_x2]
                        + sol_curr[idx_x1, idx_x2 - 1]
                    )
                    / self._grid_x2.stepsize**2
                )

                self._snapshot[idx_x1, idx_x2] = rhs
                yield lhs, rhs

    def _solve_bound(
        self, val_t: float
    ) -> Generator[tuple[torch.Tensor, float], None, None]:
        # not efficient, but readable & foolproof
        for idx_x1, val_x1 in self._grid_x1.step_with_index():
            for idx_x2, val_x2 in self._grid_x2.step_with_index():
                if self._grid_x1.is_on_boundary(val_x1) or self._grid_x2.is_on_boundary(
                    val_x2
                ):
                    yield torch.tensor([val_t, val_x1, val_x2]), 0.0

    def as_dataset(self, save_raw: bool = False) -> torch.utils.data.TensorDataset:
        out = self._output_dir / "dataset.torch"
        if not pathlib.Path(out).exists():
            logger.info("head2d: no saved data found, building data")

            self.solve()
            dataset = self.save_dataset(out)
            if save_raw:
                self.save_raw()

            logger.info("head2d: dataset saved; returning it now")
            return dataset
        else:
            logger.info(f"head2d: dataset found at {out}; returning it now")
            return torch.load(out)

    def save_raw(self) -> None:
        for tensor, f in zip([self._lhss, self._rhss], ["lhss", "rhss"]):
            filename = f"{f}.torch"
            if not pathlib.Path(filename).exists():
                torch.save(tensor, self._output_dir / filename)

    def save_dataset(self, out: pathlib.Path) -> torch.utils.data.TensorDataset:
        dataset = torch.utils.data.TensorDataset(
            torch.stack(self._lhss), torch.tensor(self._rhss).view(-1, 1)
        )
        torch.save(dataset, out)
        return dataset

    def _to_frame(self, lhss: torch.Tensor, rhss: torch.Tensor) -> torch.Tensor:
        res = torch.empty((self._grid_x1.n_pts, self._grid_x2.n_pts))

        for lhs, rhs in zip(lhss, rhss):
            res[self._grid_x1.index_of(lhs[1]), self._grid_x2.index_of(lhs[2])] = rhs
        return res

    def _plot_snapshot(self, timestep: int) -> None:
        timestep_formatted = self._grid_time.timestep_formatted(timestep)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            self._grid_x1.step(),
            self._grid_x2.step(),
            self._snapshot,
            cmap="viridis",
        )
        ax.set_xlim(*self._grid_x1._boundaries)
        ax.set_ylim(*self._grid_x2._boundaries)
        ax.set_zlim(0, 120)
        ax.set_title(
            "Heat [time-step " f"{timestep_formatted}/{self._grid_time.n_pts}" "]"
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

    pde = PDEHeat(
        GridTime(n_pts=100, stepsize=0.01),
        Grid(n_pts=50, stepsize=0.1, start=0.0),
        Grid(n_pts=50, stepsize=0.1, start=0.0),
    )
    pde.as_dataset()
