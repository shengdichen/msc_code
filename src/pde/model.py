import abc
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
        dataset_train: dataset_pde.DatasetMasked,
        datasets_eval: typing.Optional[
            typing.Sequence[dataset_pde.DatasetMasked]
        ] = None,
    ):
        self._network = network
        self._device = DEFINITION.device_preferred
        self._network.network.to(self._device)

        self._optimizer: torch.optim.optimizer.Optimizer
        self._scheduler: torch.optim.lr_scheduler.LRScheduler

        self._dataset_train = dataset_train
        self._datasets_eval = datasets_eval or []

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
    def datasets_eval(self) -> typing.Sequence[dataset_pde.DatasetMasked]:
        return self._datasets_eval

    @datasets_eval.setter
    def datasets_eval(self, value: typing.Sequence[dataset_pde.DatasetMasked]) -> None:
        self._datasets_eval = value

    def train(
        self,
        n_epochs_max: int = 1000,
        batch_size: int = 30,
        n_epochs_stale_max: int = 30,
        n_remasks_stale_max: int = 3,
        freq_report: int = 100,
        adhoc: bool = False,
    ) -> tuple[np.ndarray, list[float], tuple[list[int], list[float]]]:
        if adhoc:
            path = pathlib.Path(f"{self.path_with_remask(n_epochs_stale_max)}.pth")
        else:
            path = pathlib.Path(f"{self._path_network}.pth")

        if path.exists() and not adhoc:
            logger.info(f"model> already done! [{path}]")
            self._network.load(path)
            return np.array([]), [], ([], [])

        logger.info(f"model> learning... [{path}]")
        self._update_optimizer_scheduler(n_epochs_max)

        error_curr = math.inf
        error_best = error_curr
        epoch_best = -1
        n_epochs_stale = 0
        n_remasks_stale = 0

        errors = []
        epochs_remask = []

        record_bests = False
        bests: tuple[list[int], list[float]] = [], []

        for epoch in tqdm(range(n_epochs_max)):
            if n_epochs_stale_max > 0 and n_epochs_stale == n_epochs_stale_max:
                n_remasks_stale += 1
                logger.info(
                    f"train/stale> "
                    f"n_epochs, n_remasks: ({n_epochs_stale}, {n_remasks_stale})"
                )
                epochs_remask.append(epoch)
                record_bests = True
                if n_remasks_stale == n_remasks_stale_max:
                    logger.info(
                        "train/stale> "
                        f"giving up [best epoch: {epoch_best}; current: {epoch}"
                    )
                    break
                self._dataset_train.remask()
                n_epochs_stale = 0

            error_train = self._train_epoch(batch_size=batch_size)

            error_curr = self.eval()
            errors.append([error_train, error_curr])
            if error_curr < error_best:
                if record_bests:
                    bests[0].append(epoch)
                    bests[1].append(error_curr)
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

        return np.array(errors), epochs_remask, bests

    def path_with_remask(self, n_epochs_stale_max: int) -> pathlib.Path:
        return pathlib.Path(f"{self._path_network}--remask_{n_epochs_stale_max}")

    def _update_optimizer_scheduler(self, n_epochs_max: int) -> None:
        self._optimizer = torch.optim.Adam(self._network.network.parameters())
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer,
            step_size=max(n_epochs_max // 20, 20),
            gamma=0.5,  # more conservative decay than default
        )

    @abc.abstractmethod
    def _train_epoch(self, batch_size: int = 30) -> typing.Any:
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self, batch_size: int = 30, print_result: bool = False) -> float:
        raise NotImplementedError

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


class ModelSingle(Model):
    def __init__(
        self,
        network: factory.Network,
        dataset_train: dataset_pde.DatasetMaskedSingle,
        datasets_eval: typing.Optional[
            typing.Sequence[dataset_pde.DatasetMaskedSingle]
        ] = None,
    ):
        super().__init__(network, dataset_train, datasets_eval)

    def _train_epoch(self, batch_size: int = 30) -> float:
        self._network.network.train()
        errors = []
        for dst in self._distances_dataset(
            self._dataset_train.dataset_masked, batch_size=batch_size
        ):
            dst.mse().backward()
            errors.append(dst.mse_relative().item())
            self._optimizer.step()
            self._optimizer.zero_grad()
        self._scheduler.step()
        return np.average(errors).item()

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
        clip_at_max: typing.Optional[float] = None,
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
        errors_np = np.array(errors)
        if clip_at_max:
            errors_np = np.clip(errors_np, a_min=0.0, a_max=clip_at_max)
        if in_percentage:
            return 100 * errors_np
        return errors_np

    def _distances_dataset(
        self, dataset: T_DATASET, batch_size: int = 30
    ) -> collections.abc.Generator[distance.Distance, None, None]:
        for __, rhss_theirs, rhss_ours in self._iterate_dataset(
            dataset, batch_size=batch_size
        ):
            yield distance.Distance(rhss_ours, rhss_theirs)

    def reconstruct(self) -> tuple[np.ndarray, float]:
        batch = next(
            self._iterate_dataset(self._datasets_eval[0].dataset_masked, batch_size=30)
        )
        channels_masked, channels_truth, channels_ours = (
            batch[0][0],
            batch[1][0],
            batch[2][0],
        )

        channels = []
        for chan_masked, chan_truth, chan_ours in zip(
            channels_masked, channels_truth, channels_ours
        ):
            error = chan_ours - chan_truth
            channels.append(
                [
                    chan_truth.detach().cpu().numpy(),
                    chan_masked.detach().cpu().numpy(),
                    chan_ours.detach().cpu().numpy(),
                    error.detach().cpu().numpy(),
                ]
            )
            dst = distance.Distance(chan_ours, chan_truth)
            dst_mse = dst.mse_relative()
            print(dst.mse(), dst_mse)
        return np.array(channels), dst_mse.item()


class ModelDouble(Model):
    def __init__(
        self,
        network: factory.Network,
        dataset_train: dataset_pde.DatasetMaskedDouble,
        datasets_eval: typing.Optional[
            typing.Sequence[dataset_pde.DatasetMaskedSingle]
        ] = None,
        weight_secondary_channel: float = 0.7,
    ):
        super().__init__(network, dataset_train, datasets_eval)

        self._weights = np.array([1, weight_secondary_channel]) / (
            1 + weight_secondary_channel
        )

    def _train_epoch(self, batch_size: int = 30) -> float:
        self._network.network.train()
        errors = []
        for distances in self._distances_dataset(
            self._dataset_train.dataset_masked, batch_size=batch_size
        ):
            self._mse_abs(distances).backward()
            errors.append(self._mse_rel(distances))
            self._optimizer.step()
            self._optimizer.zero_grad()
        self._scheduler.step()
        return np.average(errors).item()

    def _mse_abs(
        self, distances: tuple[distance.Distance, distance.Distance]
    ) -> torch.Tensor:
        return (
            self._weights[0] * distances[0].mse()
            + self._weights[1] * distances[1].mse()
        )

    def _mse_rel(
        self, distances: tuple[distance.Distance, distance.Distance]
    ) -> torch.Tensor:
        return (
            self._weights[0] * distances[0].mse_relative().item()
            + self._weights[1] * distances[1].mse_relative().item()
        )

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

        abss_all, rels_all = self._eval_errors(batch_size)

        rels_all_avg = np.average(np.array(list(rels_all.values())), axis=0)
        rels_avg = np.dot(self._weights, rels_all_avg)

        if print_result:
            with np.printoptions(precision=4):
                print("-" * 10)
                for k, v in rels_all.items():
                    print(f"eval/{k}> mse: {abss_all[k]}, mse%: {v * 100}")
                abss_all_avg = np.average(np.array(list(abss_all.values())), axis=0)
                abss_avg = np.dot(self._weights, abss_all_avg)
                print(
                    "\neval/average> "
                    f"{abss_avg:.4e}, {rels_avg:.4%} [mse, mse%]"
                    f"\n{'-' * 10}"
                )

        return rels_avg

    def errors(
        self,
        in_percentage: bool = True,
        batch_size: int = 30,
        clip_at_max: typing.Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        __, rels_all_dict = self._eval_errors(batch_size)

        errors = np.array(list(rels_all_dict.values()))

        if clip_at_max:
            errors = np.clip(errors, a_min=0.0, a_max=clip_at_max)
        if in_percentage:
            errors = 100 * errors
        return errors[:, 0], errors[:, 1]

    def _eval_errors(
        self, batch_size: int = 30
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if not self._datasets_eval:
            raise ValueError(
                "no eval-dataset(s) provided "
                f"[train: {self._dataset_train.name_human()}]"
            )

        self._network.network.eval()

        abss_all, rels_all = {}, {}
        with torch.no_grad():
            for dataset in self._datasets_eval:
                abss, rels = np.empty((0, 2)), np.empty((0, 2))
                for dst_0, dst_1 in self._distances_dataset(
                    dataset.dataset_masked, batch_size=batch_size
                ):
                    abs_curr = np.array((dst_0.mse().item(), dst_1.mse().item()))
                    abss = np.concatenate((abss, np.expand_dims(abs_curr, 0)))
                    rel_curr = np.array(
                        (dst_0.mse_relative().item(), dst_1.mse_relative().item())
                    )
                    rels = np.concatenate((rels, np.expand_dims(rel_curr, 0)))

                name = dataset.name_human()
                abss_all[name] = np.average(abss, axis=0)
                rels_all[name] = np.average(rels, axis=0)
        return abss_all, rels_all

    def _distances_dataset(
        self, dataset: T_DATASET, batch_size: int = 30
    ) -> collections.abc.Generator[
        tuple[distance.Distance, distance.Distance], None, None
    ]:
        for __, rhss_theirs, rhss_ours in self._iterate_dataset(
            dataset, batch_size=batch_size
        ):
            yield (
                distance.Distance(rhss_ours[0], rhss_theirs[0]),
                distance.Distance(rhss_ours[1], rhss_theirs[1]),
            )
