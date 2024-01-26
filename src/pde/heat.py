import logging
import os
import pathlib
from typing import Generator

import matplotlib.pyplot as plt
import torch

from src.definition import DEFINITION
from src.util import grid
from src.util.gif import MakerGif

logger = logging.getLogger(__name__)


class PDEHeat:
    def __init__(
        self,
        grid_time: grid.GridTime,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        alpha: float = 0.01,
    ):
        self._grid_time, self._grid_x1, self._grid_x2 = grid_time, grid_x1, grid_x2

        self._lhss: list[torch.Tensor] = []
        self._rhss: list[float] = []
        self._snapshot = torch.zeros((self._grid_x1.n_pts, self._grid_x2.n_pts))

        self._alpha = alpha

        self._output_dir = (
            DEFINITION.BIN_DIR / f"pde/heat-"
            f"{grid_time.n_pts}_{grid_time._stepsize}-"
            f"{grid_x1.n_pts}_{grid_x1._stepsize}-"
            f"{grid_x2.n_pts}_{grid_x2._stepsize}"
        )
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
        grid.GridTime(n_pts=100, stepsize=0.01),
        grid.Grid(n_pts=50, stepsize=0.1, start=0.0),
        grid.Grid(n_pts=50, stepsize=0.1, start=0.0),
    )
    pde.as_dataset()
