import collections
import logging
import math
import pathlib
import typing

import numpy as np
import torch
from tqdm import tqdm

from src.deepl import factory
from src.definition import DEFINITION, T_DATASET
from src.numerics import distance
from src.pde import dataset as dataset_pde

logger = logging.getLogger(__name__)


class Model:
    def __init__(
        self,
        network: factory.Network,
        dataset_train: dataset_pde.DatasetMaskedSingle,
        datasets_eval: typing.Sequence[dataset_pde.DatasetMaskedSingle],
    ):
        self._network = network
        self._device = DEFINITION.device_preferred
        self._network.network.to(self._device)

        self._dataset_train = dataset_train
        self._datasets_eval = datasets_eval

        self._path_network = pathlib.Path(
            f"{str(self._dataset_train.path)}--{self._network.name}"
        )

    def load_network(self) -> None:
        path = pathlib.Path(str(self._path_network) + ".pth")
        if not path.exists():
            logger.info(f"model> learning... [{path}]")
            self.train()
        else:
            logger.info(f"model> already done! [{path}]")
            self._network.load(path)

    @property
    def datasets_eval(self) -> typing.Sequence[dataset_pde.DatasetMaskedSingle]:
        return self._datasets_eval

    @datasets_eval.setter
    def datasets_eval(
        self, value: typing.Sequence[dataset_pde.DatasetMaskedSingle]
    ) -> None:
        self._datasets_eval = value

    def train(
        self,
        n_epochs: int = 1000,
        batch_size: int = 30,
        freq_remask: int = 200,
        freq_report: int = 100,
    ) -> None:
        path = pathlib.Path(str(self._path_network) + ".pth")
        if path.exists():
            logger.info(f"model> already done! [{path}]")
            self._network.load(path)
            return

        logger.info(f"model> learning... [{path}]")

        optimizer = torch.optim.Adam(self._network.network.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(n_epochs // 20, 20),
            gamma=0.5,  # more conservative decay than default
        )

        error_curr = math.inf
        error_best = error_curr
        epoch_best = -1

        for epoch in tqdm(range(n_epochs)):
            if epoch > 0 and epoch % freq_remask == 0:
                self._dataset_train.remask()

            self._network.network.train()
            for dst in self._distances_dataset(
                self._dataset_train.dataset_masked, batch_size=batch_size
            ):
                dst.mse().backward()
                error_train = dst.mse_relative().item()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

            error_curr = self.eval()
            if error_curr < error_best:
                self._network.save(path)
                error_best = error_curr
                epoch_best = epoch

            if epoch % freq_report == 0:
                logger.info(
                    "train/curr> "
                    f"{error_train:.4%}, {error_curr:.4%} [mse: (train, eval)]"
                )
                logger.info(f"train/best> {error_best:.4%} @ epoch {epoch_best}")

    def eval(
        self,
        batch_size: int = 30,
        print_result: bool = False,
    ) -> float:
        self._network.network.eval()

        abss_all, rels_all = {}, {}
        with torch.no_grad():
            for dataset in self._datasets_eval:
                abss, rels = [], []
                for dst in self._distances_dataset(
                    dataset.dataset_masked, batch_size=batch_size
                ):
                    abss.append(dst.mse().item())
                    rels.append(dst.mse_relative().item())
                abss_all[dataset.name] = np.average(abss)
                rels_all[dataset.name] = np.average(rels)

        avg = np.average(list(rels_all.values()))

        if print_result:
            print("-" * 10)
            for k, v in rels_all.items():
                print(f"eval/{k}> {abss_all[k]:.4e}, {v:.4%} [mse, mse%]")
            print(
                "\neval/average> "
                f"{np.average(list(abss_all.values())):.4e}, {avg:.4%} [mse, mse%]"
                f"\n{'-' * 10}"
            )

        return avg

    def _distances_dataset(
        self, dataset: T_DATASET, batch_size: int = 30
    ) -> collections.abc.Generator[distance.Distance, None, None]:
        for __, rhss_theirs, rhss_ours in self._iterate_dataset(
            dataset, batch_size=batch_size
        ):
            yield distance.Distance(rhss_ours, rhss_theirs)

    def _iterate_dataset(
        self, dataset: T_DATASET, batch_size: int = 30
    ) -> collections.abc.Generator[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None
    ]:
        for lhss, rhss_theirs in torch.utils.data.DataLoader(
            dataset, batch_size=batch_size
        ):
            # NOTE:
            #   yes, we really need to coerce single-precision (cf. pytorch's conv2d)
            lhss = lhss.to(device=self._device, dtype=torch.float)
            rhss_theirs = rhss_theirs.to(device=self._device, dtype=torch.float)
            yield lhss, rhss_theirs, self._network.network(lhss)
