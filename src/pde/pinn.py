import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.pde.pde import Distance
from src.util.multidiff import MultidiffNetwork

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

    def u_d0(self) -> float:
        return 1

    def u_d1(self) -> float:
        return 0


class NetworkDense(torch.nn.Module):
    def __init__(
        self,
        size_hidden: int,
        n_hiddens: int,
        activation=torch.nn.Tanh,
    ):
        super().__init__()

        self._size_hidden, self._n_hiddens = size_hidden, n_hiddens
        self._activation = activation

        self._layers = self._make_layers()

    def _make_layers(self) -> torch.nn.Module:
        layers = []

        layers.append(torch.nn.Linear(1, self._size_hidden))
        for _ in range(self._n_hiddens):
            layers.append(torch.nn.Linear(self._size_hidden, self._size_hidden))
        layers.append(torch.nn.Linear(self._size_hidden, 1))

        return torch.nn.ModuleList(layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        res = self._activation()(self._layers[0](input))
        for layer in self._layers[1:-1]:
            res = self._activation()(layer(res))
        return self._layers[-1](res)


class Pinn:
    def __init__(self):
        torch.manual_seed(123)
        self._network = NetworkDense(32, 2)
        logger.info(f"network\n{self._network}")

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

        self._gridpts_test = torch.linspace(0, 1, 300).reshape(-1, 1)
        self._solver_exact = SolverExact(constant_delta, constant_omega_0)
        self._u_exact = self._solver_exact.eval_pde(self._gridpts_test)

    def train(self) -> None:
        for timestep in range(15001):
            self._train_step(timestep)

    def _train_step(self, timestep: int) -> None:
        self._optimiser.zero_grad()

        loss_d0, loss_d1 = self._loss_boundary_time()
        loss_pde = self._loss_pde()
        loss = loss_d0 + self._loss_lambda_1 * loss_d1 + self._loss_lambda_2 * loss_pde

        loss.backward()
        self._optimiser.step()

        if timestep % 500 == 0:
            msg = (
                f"step: {timestep:05}"
                " | "
                f"loss: {loss:.4}"
                " [(d0, d1, pd) = "
                f"({loss_d0:5.4}, {loss_d1:5.4}, {loss_pde:5.4})"
                "]"
            )
            logger.info(msg)
            if timestep % 5000 == 0:
                logger.info("plotting")
                self._plot_progress(timestep)

    def _loss_boundary_time(self) -> tuple[torch.Tensor, torch.Tensor]:
        mdn = MultidiffNetwork(self._network, self._gridpts_boundary)

        return (
            Distance(mdn.diff_0(), self._solver_exact.u_d0()).mse(),
            Distance(mdn.diff(0, 1), self._solver_exact.u_d1()).mse(),
        )

    def _loss_pde(self) -> torch.Tensor:
        mdn = MultidiffNetwork(self._network, self._gridpts_pde)
        network_pde = (
            self._constant_k * mdn.diff_0()
            + self._constant_mu * mdn.diff(0, 1)
            + mdn.diff(0, 2)
        )
        return Distance(network_pde, 0).mse()

    def _plot_progress(self, timestep: int) -> None:
        prediction = self._network(self._gridpts_test).detach()

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
            self._gridpts_test[:, 0],
            self._u_exact[:, 0],
            label="Exact solution",
            color="tab:grey",
            alpha=0.6,
        )
        plt.plot(
            self._gridpts_test[:, 0],
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
