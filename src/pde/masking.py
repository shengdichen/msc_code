import logging
from collections.abc import Callable, Iterable

import torch

from src.pde.dataset import DatasetPde, Filter
from src.pde.multidiff import MultidiffNetwork
from src.pde.network import Network
from src.pde.pde import Distance, Grid, PDEPoisson
from src.pde.saveload import SaveloadTorch

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
            if len(lhss) > 0 and len(rhss) > 0:
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


class SolverPoisson:
    def __init__(
        self,
        network: torch.nn.Module,
        batch_boundary: list[torch.Tensor],
        batch_internal: list[torch.Tensor],
    ):
        self._network = network
        self._optimiser = torch.optim.Adam(self._network.parameters())

        self._batch_boundary, self._batch_internal = batch_boundary, batch_internal
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

    def inspect(self) -> dict[str, str]:
        mode_to_percentage: dict[str, str] = {}
        for mode, (lhss, rhss) in zip(
            ["boundary", "internal"], [self._batch_boundary, self._batch_internal]
        ):
            if len(lhss) > 0 and len(rhss) > 0:
                mode_to_percentage[mode] = Distance(
                    self._eval_network(lhss), rhss
                ).mse_percentage()
            else:
                mode_to_percentage[mode] = f"0 (no {mode}-datapts)"
        return mode_to_percentage


class Masking:
    def __init__(self):
        self._grid_x1 = Grid(n_pts=50, stepsize=0.1, start=0.0)
        self._grid_x2 = Grid(n_pts=50, stepsize=0.1, start=0.0)

        self._dataset_boundary, self._dataset_internal = PDEPoisson(
            self._grid_x1, self._grid_x2, as_laplace=True
        ).as_dataset()

        self._boundary_full = MultiEval.one_big_batch(self._dataset_boundary)
        self._internal_full = MultiEval.one_big_batch(self._dataset_internal)

    def train(self) -> None:
        perc_x1, perc_x2 = (0.1, 0.1), (0.1, 0.1)
        boundary_masked, internal_masked = self._make_batch_masked(perc_x1, perc_x2)

        saveload = SaveloadTorch("poisson")
        location = saveload.rebase_location(f"network-{perc_x1}-{perc_x2}")

        def make_network() -> torch.nn.Module:
            network = Network(dim_x=2, with_time=False)
            SolverPoisson(network, boundary_masked, internal_masked).train()
            return network

        network = saveload.load_or_make(location, make_network)

        inspect_mask = SolverPoisson(
            network, boundary_masked, internal_masked
        ).inspect()
        inspect_full = SolverPoisson(
            network, self._boundary_full, self._internal_full
        ).inspect()
        logger.info(f"error (mask) {inspect_mask}")
        logger.info(f"error (full) {inspect_full}")

    def _make_batch_masked(
        self, perc_x1: tuple[float, float], perc_x2: tuple[float, float]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        range_x1 = self._make_range(self._grid_x1, *perc_x1)
        range_x2 = self._make_range(self._grid_x2, *perc_x2)

        boundary, internal = Filter(
            DatasetPde.from_datasets(self._dataset_boundary, self._dataset_internal)
        ).filter(range_x1, range_x2)

        return [boundary.lhss, boundary.rhss], [internal.lhss, internal.rhss]

    def _make_range(
        self, grid: Grid, from_start: float, to_end: float
    ) -> tuple[float, float]:
        min = grid.start + from_start * grid.n_pts * grid.stepsize
        max = grid.end - to_end * grid.n_pts * grid.stepsize
        return min, max


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )

    torch.manual_seed(42)
    Masking().train()
