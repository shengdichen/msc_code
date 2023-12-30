import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def exact_solution(d: float, w0: float, t: torch.Tensor) -> torch.Tensor:
    if d >= w0:
        raise ValueError

    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    cos = torch.cos(phi + w * t)
    exp = torch.exp(-d * t)

    return exp * 2 * A * cos


class NetworkFullConnect(nn.Module):
    def __init__(
        self,
        size_hidden: int,
        n_layers: int,
        size_input: int = 1,
        size_output: int = 1,
        activation=nn.Tanh,
    ):
        super().__init__()

        self._layer_start = nn.Sequential(
            nn.Linear(size_input, size_hidden), activation()
        )
        self._layers_hidden = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(size_hidden, size_hidden), activation())
                for _ in range(n_layers - 1)
            ]
        )
        self._layer_end = nn.Linear(size_hidden, size_output)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._layer_end(self._layers_hidden(self._layer_start(input)))


class Pinn:
    def __init__(self):
        torch.manual_seed(123)
        self._network = NetworkFullConnect(32, 3)
        self._optimiser = torch.optim.Adam(self._network.parameters(), lr=1e-3)

        # define boundary points, for the boundary loss
        self._gridpts_boundary = (
            torch.tensor(0.0).reshape(-1, 1).requires_grad_(True)
        )  # (1, 1)

        # define training points over the entire domain, for the physics loss
        self._gridpts = (
            torch.linspace(0, 1, 30).reshape(-1, 1).requires_grad_(True)
        )  # (30, 1)

        constant_d, constant_w0 = 2, 20
        self._constant_mu, self._constant_k = 2 * constant_d, constant_w0**2
        self._t_test = torch.linspace(0, 1, 300).reshape(-1, 1)
        self._u_exact = exact_solution(constant_d, constant_w0, self._t_test)

    def train(self) -> None:
        for timestep in range(15001):
            u, dudt, d2udt2 = self._train_step()

            if timestep % 5000 == 0:
                msg = (
                    f"pred_full: {u.abs().mean().item()}",
                    f"u_dt1: {dudt.abs().mean().item()}"
                    f"u_dt2: {d2udt2.abs().mean().item()}",
                )
                logger.info(msg)
                self._plot_progress(timestep)

    def _train_step(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._optimiser.zero_grad()

        # compute each term of the PINN loss function above
        # using the following hyperparameters
        lambda1, lambda2 = 1e-1, 1e-4

        # compute boundary loss (dim == (1, 1))
        pred_boundary = self._network(self._gridpts_boundary)
        loss1 = (torch.squeeze(pred_boundary) - 1) ** 2
        # dim == (1, 1)
        u_dt1 = torch.autograd.grad(
            pred_boundary,
            self._gridpts_boundary,
            torch.ones_like(pred_boundary),
            create_graph=True,
        )[0]
        u_dt2 = (torch.squeeze(u_dt1) - 0) ** 2

        # compute physics loss (dim == (30, 1))
        pred_full = self._network(self._gridpts)  # (30, 1)
        logger.info(f"shape {pred_full.shape}")
        u_dt1 = torch.autograd.grad(
            pred_full, self._gridpts, torch.ones_like(pred_full), create_graph=True
        )[0]
        # (30, 1)
        d2udt2 = torch.autograd.grad(
            u_dt1, self._gridpts, torch.ones_like(u_dt1), create_graph=True
        )[0]
        loss3 = torch.mean(
            (d2udt2 + self._constant_mu * u_dt1 + self._constant_k * pred_full) ** 2
        )

        # backpropagate joint loss, take optimiser step
        loss = loss1 + lambda1 * u_dt2 + lambda2 * loss3
        loss.backward()
        self._optimiser.step()

        return pred_full, u_dt1, d2udt2

    def _dt(self, input: torch.Tensor, times: int) -> None:
        pass

    def _plot_progress(self, timestep: int) -> None:
        prediction = self._network(self._t_test).detach()
        logger.info(
            f"prediction [step {timestep}]> "
            f"{prediction.abs().mean()} @ {prediction.shape}"
        )

        plt.figure(figsize=(6, 2.5))
        plt.scatter(
            self._gridpts.detach()[:, 0],
            torch.zeros_like(self._gridpts)[:, 0],
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
