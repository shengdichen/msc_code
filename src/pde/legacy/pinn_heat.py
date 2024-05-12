import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.deepl.network import Network
from src.numerics import distance, grid
from src.numerics.multidiff import MultidiffNetwork
from src.pde import heat

logger = logging.getLogger(__name__)


class SolverHeat2d:
    def __init__(
        self,
        loss_weight_init: float = 5.0,
        loss_weight_boundary: float = 5.0,
        loss_weight_pde: float = 1.0,
    ):
        torch.manual_seed(42)

        self._savename = pathlib.Path("../pinn_heat-final.torch")
        if self._savename.exists():
            self._network = torch.load(self._savename)
        else:
            self._network = Network(dim_x=2)

        self._optimiser = torch.optim.Adam(self._network.parameters())

        self._grid_time = grid.GridTime(n_pts=100, stepsize=0.01)
        self._grid_x1 = grid.Grid(n_pts=50, stepsize=0.1, start=0.0)
        self._grid_x2 = grid.Grid(n_pts=50, stepsize=0.1, start=0.0)
        self._grid_space = grid.Grids([self._grid_x1, self._grid_x2])

        self._dataset = heat.PDEHeat(
            self._grid_time, self._grid_x1, self._grid_x2
        ).as_dataset()
        self._dataloader = torch.utils.data.DataLoader(self._dataset, batch_size=20)

        self._lhss, self._rhss = [], []
        for lhs, rhs in self._dataset:
            self._lhss.append(lhs)
            self._rhss.append(rhs)

        self._loss_weight_init, self._loss_weight_boundary, self._loss_weight_pde = (
            loss_weight_init,
            loss_weight_boundary,
            loss_weight_pde,
        )
        self._lhss_init: list[torch.Tensor] = []
        self._rhss_init: list[torch.Tensor] = []
        self._lhss_boundary: list[torch.Tensor] = []
        self._rhss_boundary: list[torch.Tensor] = []
        self._lhss_pde: list[torch.Tensor] = []
        self._rhss_pde: list[torch.Tensor] = []
        for lhss, rhss in self._dataloader:
            self._sort_data(lhss, rhss)

    def train(self) -> None:
        logger.info(
            f"training with: "
            f"{len(self._lhss_init)} init-pts; "
            f"{len(self._lhss_boundary)} boundary-pts; "
            f"{len(self._lhss_pde)} pde-pts; "
        )

        for epoch in range(5000):
            f_epoch = f"{epoch:04}"
            loss = self._train_epoch()
            if epoch % 100 == 0:
                logger.info(f"epoch {f_epoch}> loss: {loss}")
                self.inspect()
                torch.save(self._network, f"pinn_heat-{f_epoch}.torch")

    def _train_epoch(self) -> torch.Tensor:
        self._optimiser.zero_grad()
        loss = self._calc_loss()
        loss.backward()
        self._optimiser.step()

        return loss

    def _sort_data(self, lhss: torch.Tensor, rhss: torch.Tensor) -> torch.Tensor:
        # TODO:
        #   check number of inits/boundaries/pdes
        for lhs, rhs in zip(lhss, rhss):
            # print(lhs, lhs[0], lhs[1:])
            if self._grid_time.is_init(lhs[0]):
                # print("is init")
                self._lhss_init.append(lhs)
                self._rhss_init.append(rhs)
            elif self._grid_space.is_on_boundary(lhs[1:]):
                # print("is boundary")
                self._lhss_boundary.append(lhs)
                self._rhss_boundary.append(rhs)
            else:
                # print("internal")
                self._lhss_pde.append(lhs)
                self._rhss_pde.append(rhs)

    def _eval_network(self, lhss: torch.Tensor) -> torch.Tensor:
        alpha = 0.01

        mdn = MultidiffNetwork(self._network, lhss, ["t", "x1", "x2"])
        diff_t = mdn.diff("t", 1)
        diff_x1x1 = mdn.diff("x1", 2)
        diff_x2x2 = mdn.diff("x2", 2)

        return diff_t - alpha * (diff_x1x1 + diff_x2x2)

    def _calc_loss(self) -> torch.Tensor:
        loss = torch.tensor([0.0])
        losses = []
        for lhss, rhss, weight in [
            [self._lhss_init, self._rhss_init, self._loss_weight_init],
            [
                self._lhss_boundary,
                self._rhss_boundary,
                self._loss_weight_boundary,
            ],
            [self._lhss_pde, self._rhss_pde, self._loss_weight_pde],
        ]:
            curr = (
                weight
                * distance.Distance(
                    self._eval_network(torch.stack(lhss)),
                    torch.stack(rhss),
                ).mse()
            )
            losses.append(curr)

            loss += curr

        return loss

    def inspect(self) -> None:
        mse_pecentage = (
            distance.Distance(
                self._eval_network(torch.stack(self._lhss)),
                torch.stack(self._rhss),
            ).mse_relative()
            * 100  # convert float to percentage
        )
        logger.info(f"mse-relative: {mse_pecentage}")

    def visualize(self) -> None:
        time_to_space: dict[int, torch.Tensor] = {}
        time_to_rhs: dict[int, torch.Tensor] = {}

        for lhs, rhs in self._dataset:
            time = lhs[0].item()
            idx = self._grid_time.index_of(time)
            time_to_space.setdefault(idx, [])
            time_to_rhs.setdefault(idx, [])
            time_to_space[idx].append(lhs[1:])
            time_to_rhs[idx].append(rhs)

        for time in time_to_space:
            time_to_space[time] = torch.stack(time_to_space[time])
        for time in time_to_rhs:
            time_to_rhs[time] = torch.stack(time_to_rhs[time])

        for time in time_to_space:
            self._plot_snapshot(time)

    def _plot_snapshot(self, pt_time: int) -> None:
        snapshot = np.zeros((self._grid_x1.n_pts, self._grid_x2.n_pts))
        for idx_x1, val_x1 in self._grid_x1.step_with_index():
            for idx_x2, val_x2 in self._grid_x2.step_with_index():
                snapshot[idx_x1, idx_x2] = self._eval_network(
                    torch.tensor([pt_time, val_x1, val_x2]).view(1, -1)
                )
        timestep_formatted = self._grid_time.timestep_formatted(pt_time)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            self._grid_x1.step(),
            self._grid_x2.step(),
            snapshot,
            cmap="viridis",
        )
        ax.set_xlim(self._grid_x1.start, self._grid_x1.end)
        ax.set_ylim(self._grid_x2.start, self._grid_x2.end)
        ax.set_zlim(0, 120)
        ax.set_title(
            "Heat [time-step " f"{timestep_formatted}/{self._grid_time.n_pts}" "]"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Temperature")
        plt.tight_layout()

        plt.savefig(f"./heat/frame_{timestep_formatted}")
        plt.close()

    def masking(self) -> None:
        for n_pts in [50, 60, 70, 80, 90, 100]:
            grid_x1 = grid.Grid(n_pts=n_pts, stepsize=0.1, start=0.0)
            grid_x2 = grid.Grid(n_pts=n_pts, stepsize=0.1, start=0.0)
            self._masking_one(grid_x1, grid_x2)

        for stepsize in [0.09, 0.08, 0.07, 0.06, 0.05]:
            grid_x1 = grid.Grid(n_pts=50, stepsize=stepsize, start=0.0)
            grid_x2 = grid.Grid(n_pts=50, stepsize=stepsize, start=0.0)
            self._masking_one(grid_x1, grid_x2)

    def _masking_one(self, grid_x1: grid.Grid, grid_x2: grid.Grid):
        logger.info("exact> solving")
        lhss: list[torch.Tensor] = []
        rhss: list[float] = []

        for lhs, rhs in heat.PDEHeat(self._grid_time, grid_x1, grid_x2).as_dataset():
            lhss.append(lhs)
            rhss.append(rhs)
        lhss = torch.stack(lhss)
        rhss = torch.tensor(rhss)
        logger.info("exact> done")

        self._network.eval()
        rhss_ours = self._eval_network(lhss).view(-1)

        mse_percentage = distance.Distance(rhss_ours, rhss).mse_relative() * 100
        logger.info(f"[{grid_x1}; {grid_x2}] loss%: {mse_percentage}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )

    solver = SolverHeat2d()
    solver.masking()
