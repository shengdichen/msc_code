import abc
import collections
import logging
import pathlib
import typing

import numpy as np
import torch

from src.definition import DEFINITION, T_DATASET
from src.numerics import grid
from src.util import dataset as dataset_util
from src.util import plot

logger = logging.getLogger(__name__)


class DatasetSplits:
    def __init__(self, dataset_full: torch.utils.data.dataset.TensorDataset):
        self._dataset_full = dataset_full
        self._n_instances = len(self._dataset_full)

    def split(
        self, n_instances_eval: int, n_instances_train: int
    ) -> tuple[
        torch.utils.data.dataset.TensorDataset, torch.utils.data.dataset.TensorDataset
    ]:
        indexes_eval, indexes_train = self._indexes_eval_train(
            n_instances_eval, n_instances_train
        )

        return (
            torch.utils.data.Subset(self._dataset_full, indexes_eval),
            torch.utils.data.Subset(self._dataset_full, indexes_train),
        )

    def _indexes_eval_train(
        self, n_instances_eval: int, n_instances_train: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # NOTE:
        # generate indexes in one call with |replace| set to |False| to guarantee strict
        # separation of train and eval datasets
        indexes = np.random.default_rng(seed=42).choice(
            self._n_instances,
            n_instances_eval + n_instances_train,
            replace=False,
        )
        return indexes[:n_instances_eval], indexes[-n_instances_train:]


class DatasetPDE:
    def __init__(self, grids: grid.Grids, base_dir: pathlib.Path = DEFINITION.BIN_DIR):
        self._grids = grids

        self._dataset: T_DATASET

        self._name: str
        self._base_dir = base_dir
        self._base_dir.mkdir(exist_ok=True)
        self._path: pathlib.Path

    @property
    def grids(self) -> grid.Grids:
        return self._grids

    @property
    def dataset(self) -> T_DATASET:
        return self._dataset

    @property
    def name(self) -> str:
        return self._name

    @property
    def base_dir(self) -> pathlib.Path:
        return self._base_dir

    @property
    def path(self) -> pathlib.Path:
        return self._path

    def load_full(self, n_instances: int) -> T_DATASET:
        self._path = self._base_dir / f"{self._name}-full_{n_instances}"
        return self._load(n_instances)

    def load_eval(self, n_instances: int) -> T_DATASET:
        self._path = self._base_dir / f"{self._name}-eval_{n_instances}"
        return self._load(n_instances)

    def load_train(self, n_instances: int) -> T_DATASET:
        self._path = self._base_dir / f"{self._name}-train_{n_instances}"
        return self._load(n_instances)

    def _load(self, n_instances: int) -> T_DATASET:
        path = pathlib.Path(str(self._path) + ".pth")
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
            self._base_dir / f"{self._name}-eval_{n_instances_eval}.pth",
            self._base_dir / f"{self._name}-train_{n_instances_train}.pth",
        )
        if not (path_eval.exists() and path_train.exists()):
            ds_eval, ds_train = DatasetSplits(
                self.load_full(n_instances_eval + n_instances_train)
            ).split(
                n_instances_eval=n_instances_eval, n_instances_train=n_instances_train
            )
            torch.save(ds_eval, path_eval)
            torch.save(ds_train, path_train)
        return torch.load(path_eval), torch.load(path_train)

    @abc.abstractmethod
    def as_dataset(self, n_instances: int) -> T_DATASET:
        raise NotImplementedError

    def solve(
        self, n_instances: int
    ) -> collections.abc.Generator[typing.Iterable[torch.Tensor]]:
        for __ in range(n_instances):
            yield self.solve_instance()

    @abc.abstractmethod
    def solve_instance(self) -> typing.Iterable[torch.Tensor]:
        raise NotImplementedError


class DatasetPDE2d(DatasetPDE):
    def __init__(self, grids: grid.Grids, base_dir: pathlib.Path = DEFINITION.BIN_DIR):
        if grids.n_dims != 2:
            raise ValueError(
                f"expected grid of 2 dimensions, got one with {grids.n_dims} instead"
            )

        super().__init__(grids, base_dir=base_dir)

        self._coords_x1, self._coords_x2 = self._grids.coords_as_mesh_torch()
        self._putil = plot.PlotUtil(self._grids)

    @abc.abstractmethod
    def as_dataset(self, n_instances: int) -> T_DATASET:
        raise NotImplementedError

    @abc.abstractmethod
    def solve_instance(self) -> typing.Iterable[torch.Tensor]:
        raise NotImplementedError


class DatasetMasked:
    N_CHANNELS_LHS: int
    N_CHANNELS_RHS: int

    def __init__(
        self,
        dataset_raw: DatasetPDE2d,
        masks: typing.Sequence[dataset_util.Masker],
    ):
        self._dataset_raw = dataset_raw
        self._masks = masks

        self._grids = self._dataset_raw.grids
        self._coords = self._grids.coords_as_mesh_torch()
        self._cos_coords = self._grids.cos_coords_as_mesh_torch()
        self._sin_coords = self._grids.sin_coords_as_mesh_torch()

        self._normalizer: dataset_util.Normalizer
        self._dataset_unmasked: T_DATASET
        self._dataset_masked: T_DATASET

        self._name: str
        self._base_dir = self._dataset_raw.base_dir
        self._path: pathlib.Path

    @property
    def dataset_masked(self) -> T_DATASET:
        return self._dataset_masked

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> pathlib.Path:
        return self._path

    def load_unmasked(self) -> None:
        path = pathlib.Path(str(self._path) + "--unmasked.pth")
        if not path.exists():
            logger.info(f"dataset/masked> making unmasked... [{path}]")
            self.make_unmasked()
            torch.save(self._dataset_unmasked, path)
        else:
            logger.info(f"dataset/masked> already done making unmasked! [{path}]")
            self._dataset_unmasked = torch.load(path)

    def make_unmasked(self) -> None:
        self._assemble_unmasked()
        self._normalize_unmasked()

    @abc.abstractmethod
    def _assemble_unmasked(self) -> None:
        raise NotImplementedError

    def _normalize_unmasked(self) -> None:
        self._normalizer = dataset_util.Normalizer.from_dataset(self._dataset_unmasked)
        self._dataset_unmasked = self._normalizer.normalize_dataset(
            self._dataset_unmasked
        )

    def load_masked(self) -> None:
        path = pathlib.Path(str(self._path) + "--masked.pth")
        if not path.exists():
            logger.info(f"dataset/masked> masking... [{path}]")
            self.mask()
            torch.save(self._dataset_masked, path)
        else:
            logger.info(f"dataset/masked> already done masking! [{path}]")
            self._dataset_masked = torch.load(path)

    @abc.abstractmethod
    def mask(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def remask(self) -> None:
        raise NotImplementedError


class DatasetMaskedSingle(DatasetMasked):
    MASK_IDX: int

    def __init__(self, dataset_raw: DatasetPDE2d, mask: dataset_util.Masker):
        super().__init__(dataset_raw, [mask])

        self._mask = mask

    def as_train(self, n_instances: int) -> None:
        self._dataset_raw.load_train(n_instances)

        self._name = f"{self._dataset_raw.path} train-{self._mask.as_name()}"
        self._path = (
            self._base_dir / f"{self._dataset_raw.path}--train_{self._mask.as_name()}"
        )
        self.load_unmasked()
        self.load_masked()

    def as_eval(self, n_instances: int) -> None:
        self._dataset_raw.load_eval(n_instances)

        self._name = f"{self._dataset_raw.name} eval-{self._mask.as_name()}"
        self._path = (
            self._base_dir / f"{self._dataset_raw.path}--eval_{self._mask.as_name()}"
        )
        self.load_unmasked()
        self.load_masked()

    @classmethod
    def evals_from_masks(
        cls,
        dataset_raw: DatasetPDE2d,
        masks: dataset_util.Masker,
        n_instances: int,
    ) -> typing.Sequence["DatasetMaskedSingle"]:
        evals = []
        for mask in masks:
            ds = cls(dataset_raw, mask)
            ds.as_eval(n_instances)
            evals.append(ds)
        return evals

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
            rhss.append(instance[self.MASK_IDX].unsqueeze(0))
        self._dataset_unmasked = torch.utils.data.TensorDataset(
            torch.stack(lhss), torch.stack(rhss)
        )

    def mask(self) -> None:
        lhss, rhss = dataset_util.DatasetPde.from_dataset(
            self._dataset_unmasked
        ).lhss_rhss
        for lhs in lhss:
            lhs[self.MASK_IDX] = self._mask.mask(lhs[self.MASK_IDX])
        self._dataset_masked = torch.utils.data.TensorDataset(lhss, rhss)

    def remask(self) -> None:
        # NOTE:
        #   we intentionally do NOT auto-save remasked dataset

        logger.info(f"dataset/masked> remasking... [with {self._mask.as_name()}]")
        self.mask()
