import abc
import pathlib
import typing

import numpy as np
import torch

from src.numerics import grid
from src.util import dataset, plot


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
    def __init__(self, grids: grid.Grids):
        self._grids = grids

    def load_full(
        self, n_instances: int, base_dir: pathlib.Path = pathlib.Path(".")
    ) -> torch.utils.data.dataset.TensorDataset:
        path = base_dir / f"{self.as_name()}-full-{n_instances}.pth"
        if not path.exists():
            torch.save(self.as_dataset(n_instances), path)
        return torch.load(path)

    def load_split(
        self,
        n_instances_eval: int,
        n_instances_train: int,
        base_dir: pathlib.Path = pathlib.Path("."),
    ) -> torch.utils.data.dataset.TensorDataset:
        path_eval, path_train = (
            base_dir / f"{self.as_name()}-eval-{n_instances_eval}.pth",
            base_dir / f"{self.as_name()}-train-{n_instances_train}.pth",
        )
        if not (path_eval.exists() and path_train.exists()):
            ds_eval, ds_train = DatasetSplits(
                self.load_full(n_instances_eval + n_instances_train, base_dir)
            ).split(
                n_instances_eval=n_instances_eval, n_instances_train=n_instances_train
            )
            torch.save(ds_eval, path_eval)
            torch.save(ds_train, path_train)
        return torch.load(path_eval), torch.load(path_train)

    @abc.abstractmethod
    def as_dataset(self, n_instances: int) -> torch.utils.data.dataset.TensorDataset:
        raise NotImplementedError

    def solve(self, n_instances: int) -> typing.Iterable[typing.Iterable[torch.Tensor]]:
        for __ in range(n_instances):
            yield self.solve_instance()

    @abc.abstractmethod
    def solve_instance(self) -> typing.Iterable[torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def as_name(self) -> str:
        raise NotImplementedError


class DatasetPDE2d(DatasetPDE):
    def __init__(self, grid_x1: grid.Grid, grid_x2: grid.Grid):
        self._grid_x1, self._grid_x2 = grid_x1, grid_x2

        super().__init__(grid.Grids([self._grid_x1, self._grid_x2]))
        self._putil = plot.PlotUtil(self._grids)

    @abc.abstractmethod
    def as_dataset(self, n_instances: int) -> torch.utils.data.dataset.TensorDataset:
        raise NotImplementedError

    @abc.abstractmethod
    def solve_instance(self) -> typing.Iterable[torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def as_name(self) -> str:
        raise NotImplementedError


class DatasetMasked:
    @abc.abstractmethod
    def make(
        self, dataset: torch.utils.data.dataset.TensorDataset
    ) -> torch.utils.data.dataset.TensorDataset:
        raise NotImplementedError

    @abc.abstractmethod
    def as_name(self) -> str:
        raise NotImplementedError
