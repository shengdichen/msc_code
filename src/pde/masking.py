import logging
from collections.abc import Callable

import torch

from src.pde.dataset import DatasetPde, Filter, MultiEval
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


class SolverPoisson:
    def __init__(
        self,
        network: torch.nn.Module,
        dataset_boundary: DatasetPde,
        dataset_internal: DatasetPde,
    ):
        self._network = network
        self._optimiser = torch.optim.Adam(self._network.parameters())

        self._dataset_boundary, self._dataset_internal = (
            dataset_boundary,
            dataset_internal,
        )
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
                    f"epoch {epoch:04}> " f"loss: {loss:.4} " f"{self.inspect()}"
                )

    def _train_epoch(self) -> float:
        self._optimiser.zero_grad()
        loss = MultiEval(self._eval_network).loss_weighted(
            [self._dataset_boundary, self._dataset_internal],
            [5.0, 1.0],
        )
        loss.backward()
        self._optimiser.step()

        return loss.item()

    def inspect(self) -> dict[str, str]:
        mode_to_percentage: dict[str, str] = {}
        for mode, dataset in zip(
            ["boundary", "internal"], [self._dataset_boundary, self._dataset_internal]
        ):
            if not dataset.is_empty():
                mode_to_percentage[mode] = Distance(
                    self._eval_network(dataset.lhss), dataset.rhss
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

        self._boundary_full = DatasetPde.from_datasets(self._dataset_boundary)
        self._internal_full = DatasetPde.from_datasets(self._dataset_internal)

    def train(self) -> None:
        for perc_x1, perc_x2 in [
            ((0.1, 0.1), (0.1, 0.1)),
            ((0.1, 0.2), (0.1, 0.1)),
            ((0.1, 0.1), (0.1, 0.2)),
            ((0.2, 0.1), (0.1, 0.1)),
            ((0.1, 0.1), (0.2, 0.1)),
        ]:
            logger.info(f"{perc_x1} x {perc_x2}")
            self._train_one(perc_x1, perc_x2)

    def _train_one(
        self, perc_x1: tuple[float, float], perc_x2: tuple[float, float]
    ) -> None:
        dataset_boundary, dataset_internal = self._make_batch_masked(perc_x1, perc_x2)

        def make_network() -> torch.nn.Module:
            network = Network(dim_x=2, with_time=False)
            SolverPoisson(network, dataset_boundary, dataset_internal).train()
            return network

        saveload = SaveloadTorch("poisson")
        network = saveload.load_or_make(
            saveload.rebase_location(f"network-{perc_x1}-{perc_x2}"), make_network
        )

        inspect_mask = SolverPoisson(
            network, dataset_boundary, dataset_internal
        ).inspect()
        inspect_full = SolverPoisson(
            network, self._boundary_full, self._internal_full
        ).inspect()
        logger.info(f"error (mask) {inspect_mask}")
        logger.info(f"error (full) {inspect_full}")

    def _make_batch_masked(
        self, perc_x1: tuple[float, float], perc_x2: tuple[float, float]
    ) -> tuple[DatasetPde, DatasetPde]:
        range_x1 = self._grid_x1.min_max_n_pts(*perc_x1)
        range_x2 = self._grid_x2.min_max_n_pts(*perc_x2)

        dataset = DatasetPde.from_datasets(
            self._dataset_boundary, self._dataset_internal
        )
        filter = Filter(dataset)
        res = filter.filter(range_x1, range_x2)

        return res


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )

    torch.manual_seed(42)
    Masking().train()
