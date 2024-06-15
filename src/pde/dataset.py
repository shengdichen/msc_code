import abc
import collections
import logging
import pathlib
import typing

import torch

from src.definition import DEFINITION, T_DATASET
from src.numerics import grid
from src.util import dataset as dataset_util
from src.util import plot

logger = logging.getLogger(__name__)


class DatasetPDE:
    N_CHANNELS: int

    def __init__(
        self,
        grids: grid.Grids,
        name_problem: str,
        name_dataset: str,
    ):
        self._grids = grids

        self._dataset: T_DATASET

        self._name_problem, self._name_dataset = name_problem, name_dataset
        self._name_human = (
            f"{self._name_problem.title()}"
            " "
            f"\"{' '.join(self._name_dataset.split('_'))}\""
        )
        self._base_dir = DEFINITION.BIN_DIR / name_problem
        self._base_dir.mkdir(exist_ok=True)
        self._path: pathlib.Path

    @property
    def grids(self) -> grid.Grids:
        return self._grids

    @property
    def dataset(self) -> T_DATASET:
        return self._dataset

    @property
    def name_problem(self) -> str:
        return self._name_problem

    @property
    def name_dataset(self) -> str:
        return self._name_dataset

    @property
    def name_human(self) -> str:
        return self._name_human

    @property
    def base_dir(self) -> pathlib.Path:
        return self._base_dir

    @property
    def path(self) -> pathlib.Path:
        return self._path

    def load_full(self, n_instances: int) -> T_DATASET:
        self._path = self._base_dir / f"{self._name_dataset}-full_{n_instances}"
        return self._load(n_instances)

    def load_eval(self, n_instances: int) -> T_DATASET:
        self._path = self._base_dir / f"{self._name_dataset}-eval_{n_instances}"
        return self._load(n_instances)

    def load_train(self, n_instances: int) -> T_DATASET:
        self._path = self._base_dir / f"{self._name_dataset}-train_{n_instances}"
        return self._load(n_instances)

    def _load(self, n_instances: int) -> T_DATASET:
        path = pathlib.Path(f"{self._path}.pth")
        if not path.exists():
            logger.info(f"dataset/raw> making... [{path}]")
            torch.save(self.as_dataset(n_instances), path)
        else:
            logger.info(f"dataset/raw> already done! [{path}]")

        self._dataset = torch.load(path)
        return self._dataset

    def load_split(
        self,
        n_instances_eval: int,
        n_instances_train: int,
    ) -> tuple[T_DATASET, T_DATASET]:
        path_eval, path_train = (
            self._base_dir / f"{self._name_dataset}-eval_{n_instances_eval}.pth",
            self._base_dir / f"{self._name_dataset}-train_{n_instances_train}.pth",
        )
        if not (path_eval.exists() and path_train.exists()):
            ds_eval, ds_train = dataset_util.Splitter(
                self.load_full(n_instances_eval + n_instances_train)
            ).split(
                n_instances_eval=n_instances_eval, n_instances_train=n_instances_train
            )
            torch.save(ds_eval, path_eval)
            torch.save(ds_train, path_train)
        return torch.load(path_eval), torch.load(path_train)

    def as_dataset(self, n_instances: int) -> T_DATASET:
        instances: list[typing.Sequence[torch.Tensor]] = list(self.solve(n_instances))
        channels: list[list[torch.Tensor]] = [[] for __ in range(len(instances[0]))]
        for instance in instances:
            for channel, item in zip(channels, instance):
                channel.append(item)
        return torch.utils.data.TensorDataset(
            *[torch.stack(channel) for channel in channels]
        )

    def solve(
        self, n_instances: int
    ) -> collections.abc.Generator[typing.Sequence[torch.Tensor], None, None]:
        for __ in range(n_instances):
            yield self.solve_instance()

    @abc.abstractmethod
    def solve_instance(self) -> typing.Sequence[torch.Tensor]:
        raise NotImplementedError


class DatasetPDE2d(DatasetPDE):
    def __init__(
        self,
        grids: grid.Grids,
        name_problem: str,
        name_dataset: str,
    ):
        if grids.n_dims != 2:
            raise ValueError(
                f"expected grid of 2 dimensions, got one with {grids.n_dims} instead"
            )

        super().__init__(grids, name_problem, name_dataset)

        self._coords_x1, self._coords_x2 = self._grids.coords_as_mesh_torch()
        self._putil = plot.PlotUtil(self._grids)

    @abc.abstractmethod
    def solve_instance(self) -> typing.Sequence[torch.Tensor]:
        raise NotImplementedError


class DatasetMasked:
    def __init__(
        self,
        dataset_raw: DatasetPDE2d,
        masks: typing.Sequence[dataset_util.Masker],
        masks_indexes: typing.Sequence[int],
        save_unmasked: bool = True,
    ):
        self._dataset_raw = dataset_raw
        self._masks = masks
        self._masks_indexes = masks_indexes

        self._grids = self._dataset_raw.grids
        self._coords = self._grids.coords_as_mesh_torch()
        self._cos_coords = self._grids.cos_coords_as_mesh_torch()
        self._sin_coords = self._grids.sin_coords_as_mesh_torch()

        self._normalizer: dataset_util.Normalizer
        self._dataset_unmasked: T_DATASET
        self._dataset_masked: T_DATASET

        # NOTE:
        #   first remask call will (re-)produce the (saved) masked dataset, thus we need
        #   to remask TWICE during this first remask call
        self._need_second_remasking: bool = True

        self._base_dir = self._dataset_raw.base_dir
        self._path: pathlib.Path
        self._save_unmasked = save_unmasked

    @property
    def grids(self) -> grid.Grids:
        return self._grids

    @property
    def dataset_masked(self) -> T_DATASET:
        return self._dataset_masked

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def masks(self) -> typing.Sequence[dataset_util.Masker]:
        return self._masks

    def name_human(self, multiline: bool = False) -> str:
        name_dataset = f"Dataset: {self._dataset_raw.name_human}"
        connector = "\n" if multiline else " "
        name_masks = f"Masking: {' + '.join([mask.name_human for mask in self._masks])}"
        return f"{name_dataset}{connector}{name_masks}"

    def as_train(self, n_instances: int) -> None:
        self._dataset_raw.load_train(n_instances)

        self._save_unmasked = True
        self._update_path()
        self.load_unmasked()
        self.load_masked()

    def as_eval(self, n_instances: int) -> None:
        self._dataset_raw.load_eval(n_instances)

        self._save_unmasked = False
        self._update_path()
        self.load_masked()

    def _update_path(self) -> None:
        self._path = (
            self._base_dir / f"{self._dataset_raw.path}"
            "--"
            f"{'-'.join([mask.name for mask in self._masks])}"
        )

    def load_unmasked(self) -> None:
        path = pathlib.Path(f"{self._path}--unmasked.pth")
        if not path.exists():
            if self._save_unmasked:
                logger.info(f"dataset/masked> making unmasked... [{path}]")
            else:
                logger.info("dataset/masked> making unmasked... [not saving]")
            self.make_unmasked()
            if self._save_unmasked:
                torch.save(self._dataset_unmasked, path)
        else:
            logger.info(f"dataset/masked> already done making unmasked! [{path}]")
            self._dataset_unmasked = torch.load(path)

    def make_unmasked(self) -> None:
        self._assemble_unmasked()
        self._normalize_unmasked()

    def _assemble_unmasked(self) -> None:
        lhss, rhss = [], []
        for instance in self._dataset_raw.dataset:
            lhss.append(
                torch.stack(
                    [
                        *instance,
                        *self._coords,
                        *self._cos_coords,
                        *self._sin_coords,
                    ]
                )
            )
            rhss.append(
                torch.stack([instance[mask_idx] for mask_idx in self._masks_indexes])
            )
        self._dataset_unmasked = torch.utils.data.TensorDataset(
            torch.stack(lhss), torch.stack(rhss)
        )

    def _normalize_unmasked(self) -> None:
        self._normalizer = dataset_util.Normalizer.from_dataset(self._dataset_unmasked)
        self._dataset_unmasked = self._normalizer.normalize_dataset(
            self._dataset_unmasked
        )

    def load_masked(self) -> None:
        path = pathlib.Path(f"{self._path}.pth")
        if not path.exists():
            logger.info(f"dataset/masked> masking... [{path}]")
            self.load_unmasked()
            self.mask()
            torch.save(self._dataset_masked, path)
        else:
            logger.info(f"dataset/masked> already done masking! [{path}]")
            self._dataset_masked = torch.load(path)

    def mask(self) -> None:
        lhss, rhss = dataset_util.DatasetPde.from_dataset(
            self._dataset_unmasked
        ).lhss_rhss
        for lhs in lhss:
            for mask_idx, mask in zip(self._masks_indexes, self._masks):
                lhs[mask_idx] = mask.mask(lhs[mask_idx])
        self._dataset_masked = torch.utils.data.TensorDataset(lhss, rhss)

    def remask(self) -> None:
        # NOTE:
        #   we intentionally do NOT auto-save remasked dataset

        logger.info(
            "dataset/masked> remasking... "
            f"[with {' + '.join([mask.name_human for mask in self._masks])}]"
        )
        self.mask()
        if self._need_second_remasking:
            self.mask()
            self._need_second_remasking = False


class DatasetMaskedSingle(DatasetMasked):
    def __init__(
        self, dataset_raw: DatasetPDE2d, mask: dataset_util.Masker, mask_index: int
    ):
        super().__init__(dataset_raw, [mask], [mask_index])

        self._mask = mask

    @classmethod
    def evals_from_masks(
        cls,
        dataset_raw: DatasetPDE2d,
        masks: typing.Sequence[dataset_util.Masker],
        mask_index: int,
        n_instances: int,
    ) -> typing.Sequence["DatasetMaskedSingle"]:
        evals = []
        for mask in masks:
            ds = cls(dataset_raw, mask, mask_index)
            ds.as_eval(n_instances)
            evals.append(ds)
        return evals


class DatasetMaskedDouble(DatasetMasked):
    def __init__(
        self,
        dataset_raw: DatasetPDE2d,
        masks: tuple[dataset_util.Masker, dataset_util.Masker],
        masks_indexes: tuple[int, int],
    ):
        super().__init__(dataset_raw, masks, masks_indexes)
