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
        datasets_eval: typing.Optional[
            typing.Sequence[dataset_pde.DatasetMaskedSingle]
        ] = None,
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
        path = pathlib.Path(f"{self._path_network}.pth")
        if not path.exists():
            logger.info(f"model> learning... [{path}]")
            self.train()
        else:
            logger.info(f"model> already done! [{path}]")
            self._network.load(path)

    @property
    def name_network(self) -> str:
        return self._network.name

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
        n_epochs_max: int = 1000,
        batch_size: int = 30,
        n_epochs_stale_max: int = 30,
        n_remasks_stale_max: int = 3,
        freq_report: int = 100,
    ) -> None:
        path = pathlib.Path(f"{self._path_network}.pth")
        if path.exists():
            logger.info(f"model> already done! [{path}]")
            self._network.load(path)
            return

        logger.info(f"model> learning... [{path}]")

        optimizer = torch.optim.Adam(self._network.network.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(n_epochs_max // 20, 20),
            gamma=0.5,  # more conservative decay than default
        )

        error_curr = math.inf
        error_best = error_curr
        epoch_best = -1
        n_epochs_stale = 0
        n_remasks_stale = 0

        for epoch in tqdm(range(n_epochs_max)):
            if n_epochs_stale == n_epochs_stale_max:
                n_remasks_stale += 1
                logger.info(
                    f"train/stale> "
                    f"n_epochs, n_remasks: ({n_epochs_stale}, {n_remasks_stale})"
                )
                if n_remasks_stale == n_remasks_stale_max:
                    logger.info(
                        "train/stale> "
                        f"giving up [best epoch: {epoch_best}; current: {epoch}"
                    )
                    break
                self._dataset_train.remask()
                n_epochs_stale = 0

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
                n_epochs_stale = 0
                n_remasks_stale = 0
            else:
                n_epochs_stale += 1
                logger.debug(
                    f"train/stale> "
                    f"n_epochs, n_remasks: ({n_epochs_stale}, {n_remasks_stale})"
                )

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
        if not self._datasets_eval:
            raise ValueError(
                "no eval-dataset(s) provided "
                f"[train: {self._dataset_train.name_human()}]"
            )

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
                name = dataset.name_human()
                abss_all[name] = np.average(abss)
                rels_all[name] = np.average(rels)

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

    def errors(
        self,
        in_percentage: bool = True,
        batch_size: int = 30,
        clip_at_max: typing.Optional[int] = None,
    ) -> np.ndarray:
        errors = []

        self._network.network.eval()
        with torch.no_grad():
            for dataset in self._datasets_eval:
                errors_curr = np.average(
                    [
                        dst.mse_relative().item()
                        for dst in self._distances_dataset(
                            dataset.dataset_masked, batch_size=batch_size
                        )
                    ]
                )
                errors.append(errors_curr)
        errors = np.array(errors)
        if clip_at_max:
            errors = np.clip(errors, a_min=0.0, a_max=clip_at_max)
        if in_percentage:
            return 100 * errors
        return errors

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
