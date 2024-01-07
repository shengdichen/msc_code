import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.pde.multidiff import Multidiff
from src.pde.pde import Distance

logger = logging.getLogger(__name__)


class SolverExact:
    def __init__(self, delta: float = 2, omega_0: float = 20):
        if delta >= omega_0:
            raise ValueError
        self._delta, self._omega_0 = delta, omega_0

    def eval_pde(self, lhs: torch.Tensor) -> torch.Tensor:
        omega = np.sqrt(self._omega_0**2 - self._delta**2)
        phi = np.arctan(-self._delta / omega)
        capital_a = 1 / (2 * np.cos(phi))

        return (
            torch.exp(-self._delta * lhs) * 2 * capital_a * torch.cos(phi + omega * lhs)
        )

    def eval_boundary_time(self, lhs: torch.Tensor) -> torch.Tensor:
        return 1


class NetworkDense(nn.Module):
    def __init__(
        self,
        size_hidden: int,
        n_layers: int,
        activation=nn.Tanh,
    ):
        super().__init__()

        self._size_layer_hidden = size_hidden
        self._n_layers_hidden = n_layers
        self._activation = activation

        self._layer_start = nn.Sequential(nn.Linear(1, size_hidden), activation())
        # hiddens = []
        # for _ in range(self._n_layers_hidden - 1):
        #     hiddens.append(
        #         nn.Sequential(
        #             nn.Linear(self._size_layer_hidden, self._size_layer_hidden),
        #             activation(),
        #         )
        #     )
        #     # hiddens.append(nn.Linear(self._size_layer_hidden, self._size_layer_hidden))
        #     # hiddens.append(self._activation())
        # self._layers_hidden = nn.Sequential(*hiddens)

        self._layers_hidden = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(size_hidden, size_hidden), activation())
                for _ in range(n_layers - 1)
            ]
        )
        self._layer_end = nn.Linear(size_hidden, 1)

        # self._layers = self._make_layers()

    def _make_layers(self) -> nn.Module:
        layers = []

        layers.append(nn.Linear(1, self._size_layer_hidden))

        layers.append(self._activation())
        for _ in range(self._n_layers_hidden):
            layers.append(nn.Linear(self._size_layer_hidden, self._size_layer_hidden))
            layers.append(self._activation())

        layers.append(nn.Linear(self._size_layer_hidden, 1))

        return nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._layer_end(self._layers_hidden(self._layer_start(input)))
        # return self._layers(input)


class Pinn:
    def __init__(self):
        torch.manual_seed(123)
        self._network = NetworkDense(32, 3)
        print(self._network)

        self._optimiser = torch.optim.Adam(self._network.parameters(), lr=1e-3)
        self._loss_lambda_1, self._loss_lambda_2 = 1e-1, 1e-4

        self._gridpts_boundary = torch.tensor(0.0, requires_grad=True).reshape(
            -1, 1
        )  # (1, 1)

        self._gridpts_pde = torch.linspace(0, 1, 30, requires_grad=True).reshape(
            -1, 1
        )  # (30, 1)

        constant_delta, constant_omega_0 = 2, 20
        self._constant_mu, self._constant_k = 2 * constant_delta, constant_omega_0**2
        self._t_test = torch.linspace(0, 1, 300).reshape(-1, 1)

        self._solver_exact = SolverExact(constant_delta, constant_omega_0)
        self._u_exact = self._solver_exact.eval_pde(self._t_test)

    def train(self) -> None:
        for timestep in range(15001):
            self._train_step(timestep)

    def _train_step(self, timestep: int) -> None:
        self._optimiser.zero_grad()

        loss_d0, loss_d1 = self._loss_boundary_time()
        loss_pde = self._loss_pde()

        if timestep % 100 == 0:
            msg = (
                f"step: {timestep}"
                " | "
                f"loss-d0: {loss_d0}"
                " | "
                f"loss-d1: {loss_d1}"
                " | "
                f"loss-pde: {loss_pde}"
            )
            logger.info(msg)
            if timestep % 5000 == 0:
                logger.info("plotting")
                self._plot_progress(timestep)

        loss = loss_d0 + self._loss_lambda_1 * loss_d1 + self._loss_lambda_2 * loss_pde
        loss.backward()
        self._optimiser.step()

    def _loss_boundary_time(self) -> tuple[torch.Tensor, torch.Tensor]:
        u_d0 = self._network(self._gridpts_boundary)  # (1, 1)
        loss_d0 = Distance(u_d0, 1).mse()

        md = Multidiff(rhs=u_d0, lhs=self._gridpts_boundary)
        u_dt1 = md.diff()
        loss_d1 = Distance(u_dt1, 0).mse()

        return loss_d0, loss_d1

    def _loss_pde(self) -> torch.Tensor:
        u_d0 = self._network(self._gridpts_pde)  # (30, 1)
        md = Multidiff(rhs=torch.sum(u_d0), lhs=self._gridpts_pde)
        u_d1t = md.diff()
        u_d2t = md.diff()

        pred = u_d2t + self._constant_mu * u_d1t + self._constant_k * u_d0
        return Distance(pred, 0).mse()

    def _plot_progress(self, timestep: int) -> None:
        prediction = self._network(self._t_test).detach()

        plt.figure(figsize=(6, 2.5))
        plt.scatter(
            self._gridpts_pde.detach()[:, 0],
            torch.zeros_like(self._gridpts_pde)[:, 0],
            s=20,
            lw=0,
            color="tab:green",
            alpha=0.6,
        )
        plt.scatter(
            self._gridpts_boundary.detach()[:, 0],
            torch.zeros_like(self._gridpts_boundary)[:, 0],
            s=20,
            lw=0,
            color="tab:red",
            alpha=0.6,
        )
        plt.plot(
            self._t_test[:, 0],
            self._u_exact[:, 0],
            label="Exact solution",
            color="tab:grey",
            alpha=0.6,
        )
        plt.plot(
            self._t_test[:, 0],
            prediction[:, 0],
            label="PINN solution",
            color="tab:green",
        )
        title = f"Training step {timestep}"
        plt.title(title)
        plt.legend()

        plt.savefig(title)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )

    pinn = Pinn()
    pinn.train()
