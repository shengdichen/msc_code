import logging
from collections.abc import Callable, Iterable

import torch

from src.pde.multidiff import MultidiffNetwork
from src.pde.network import Network
from src.pde.pde import Distance, Grid, PDEPoisson

logger = logging.getLogger(__name__)


class Learner:
    def __init__(self, epochs_train: int = 5000, frequency_report: int = 100):
        self._epochs_train = epochs_train
        self._frequency_report = frequency_report

    def train(self) -> None:
        for epoch in range(self._epochs_train):
            self._train_epoch()
            if epoch % self._frequency_report == 0:
                self._report_train()

    def _train_epoch(self) -> None:
        pass

    def _report_train(self) -> None:
        pass


class MultiEval:
    def __init__(
        self,
        eval_network: Callable[[torch.Tensor], torch.Tensor],
    ):
        self._eval_network = eval_network

    def loss_weighted(
        self, batches: Iterable[list[torch.Tensor]], weights: Iterable[float]
    ) -> torch.Tensor:
        losses = torch.tensor([0.0])
        for weight, (lhss, rhss) in zip(weights, batches):
            losses += weight * Distance(self._eval_network(lhss), rhss).mse()

        return losses

    @staticmethod
    def one_big_batch(
        dataset: torch.utils.data.dataset.TensorDataset,
    ) -> list[torch.Tensor]:
        return list(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=len(dataset),
            )
        )[0]


class Masking:
    def __init__(self):
        torch.manual_seed(42)

        grid_x1 = Grid(n_pts=50, stepsize=0.1, start=0.0)
        grid_x2 = Grid(n_pts=50, stepsize=0.1, start=0.0)

        self._network = Network(dim_x=2, with_time=False)
        self._optimiser = torch.optim.Adam(self._network.parameters())

        self._batch_boundary, self._batch_internal = [
            MultiEval.one_big_batch(dataset)
            for dataset in PDEPoisson(grid_x1, grid_x2, as_laplace=True).as_dataset()
        ]
        self._eval_network = self._make_eval_network()

    def _make_eval_network(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def f(lhss: torch.Tensor) -> torch.Tensor:
            mdn = MultidiffNetwork(self._network, lhss, ["x1", "x2"])
            return mdn.diff("x1", 2) + mdn.diff("x2", 2)

        return f

    def train(self) -> None:
        for epoch in range(5000):
            loss = self._train_epoch()
            if epoch % 100 == 0:
                logger.info(
                    f"epoch {epoch:04}> "
                    f"loss: {loss:.4}; "
                    f"[bound, intern]: {self.inspect()}"
                )

    def _train_epoch(self) -> float:
        self._optimiser.zero_grad()
        loss = MultiEval(self._eval_network).loss_weighted(
            [self._batch_boundary, self._batch_internal],
            [5.0, 1.0],
        )
        loss.backward()
        self._optimiser.step()

        return loss.item()

    def inspect(self) -> list[str]:
        percentages = []
        for lhss, rhss in [self._batch_boundary, self._batch_internal]:
            perc = (
                Distance(self._eval_network(lhss), torch.tensor(rhss, dtype=float))
                .mse_relative()
                .item()
                * 100
            )
            percentages.append(f"{perc:.4}%")
        return percentages


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )

    Masking().train()
