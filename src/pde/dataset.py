import abc
import typing

import numpy as np
import torch

from src.numerics import grid
from src.util import dataset, plot


class DatasetPDE:
    def __init__(self, grids: grid.Grids):
        self._grids = grids

    @abc.abstractmethod
    def as_dataset(self, n_instances: int) -> torch.utils.data.dataset.TensorDataset:
        raise NotImplementedError

    def solve(self, n_instances: int) -> typing.Iterable[typing.Iterable[torch.Tensor]]:
        for __ in range(n_instances):
            yield self.solve_instance()

    @abc.abstractmethod
    def solve_instance(self) -> typing.Iterable[torch.Tensor]:
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
        self, n_instances_eval: int = 300, n_instances_train: int = 100
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


class DatasetMasked:
    def __init__(
        self,
        dataset_raw: torch.utils.data.dataset.TensorDataset,
        masks: typing.Sequence[dataset.Masker],
    ):
        self._dataset_raw = dataset_raw
        self._masks = masks

    @abc.abstractmethod
    def make(self) -> torch.utils.data.dataset.TensorDataset:
        raise NotImplementedError

    @abc.abstractmethod
    def as_name(self) -> str:
        raise NotImplementedError
