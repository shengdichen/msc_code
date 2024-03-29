import abc
import itertools
import math
import typing
from collections.abc import Callable, Iterable

import numpy as np
import torch

from src.numerics import distance


class DatasetPde:
    def __init__(
        self,
        lhss: torch.Tensor,
        rhss: torch.Tensor,
    ):
        self._check_lhss_rhss(lhss, rhss)

        self._lhss, self._rhss = lhss, rhss
        self._dataset = torch.utils.data.TensorDataset(lhss, rhss)
        self._n_datapts, self._n_dims = self._lhss.shape[0], self._lhss.shape[1]

    @staticmethod
    def _check_lhss_rhss(lhss: torch.Tensor, rhss: torch.Tensor) -> None:
        if len(lhss) != len(rhss):
            raise ValueError("umatched number of lhs and rhs")
        if len(lhss.shape) != 2:
            raise ValueError("every lhs must be a tensor")
        if len(rhss.shape) != 2:
            raise ValueError(
                "every rhs must be a tensor [consider calling view(-1 ,1) on it]"
            )

    @property
    def lhss(self) -> torch.Tensor:
        return self._lhss

    @property
    def rhss(self) -> torch.Tensor:
        return self._rhss

    @property
    def dataset(self) -> torch.utils.data.dataset.TensorDataset:
        return self._dataset

    @property
    def n_datapts(self) -> int:
        return self._n_datapts

    @property
    def n_dims(self) -> int:
        return self._n_dims

    def is_empty(self) -> bool:
        return self._n_datapts == 0

    @classmethod
    def from_lhss_rhss_raw(
        cls, lhss: list[list[float]], rhss: list[float]
    ) -> "DatasetPde":
        lhss = torch.tensor(lhss, dtype=torch.float)
        rhss = torch.tensor(rhss, dtype=torch.float).view(-1, 1)
        return cls(lhss, rhss)

    @classmethod
    def from_lhss_rhss_torch(
        cls, lhss: list[torch.Tensor], rhss: list[torch.Tensor]
    ) -> "DatasetPde":
        if not lhss:
            lhss_torch, rhss_torch = torch.tensor([]), torch.tensor([])
        else:
            lhss_torch, rhss_torch = torch.stack(lhss), torch.stack(rhss).view(-1, 1)

        return cls(lhss_torch, rhss_torch)

    @classmethod
    def from_datasets(
        cls, *datasets: torch.utils.data.dataset.TensorDataset
    ) -> "DatasetPde":
        lhss: list[torch.Tensor] = []
        rhss: list[torch.Tensor] = []
        for dataset in datasets:
            for lhs, rhs in dataset:
                lhss.append(lhs)
                rhss.append(rhs)

        return cls.from_lhss_rhss_torch(lhss, rhss)

    @staticmethod
    def one_big_batch(dataset: torch.utils.data.dataset.TensorDataset) -> list:
        return list(torch.utils.data.DataLoader(dataset, batch_size=len(dataset)))[0]


class Masker:
    @abc.abstractmethod
    def mask(self, full: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MaskerRandom(Masker):
    def __init__(
        self,
        perc_to_mask: float = 0.5,
        seed: typing.Optional[int] = None,
    ):
        super().__init__()

        self._rng = np.random.default_rng(seed=seed)
        self._perc_to_mask = perc_to_mask

    def mask(self, full: torch.Tensor) -> torch.Tensor:
        res = full.detach().clone()
        for idx in self._indexes_to_mask(full):
            res[tuple(idx)] = 0
        return res

    def _indexes_to_mask(self, full: torch.Tensor) -> np.ndarray:
        n_gridpts_to_mask = int(self._perc_to_mask * np.prod(full.shape))

        indexes_full = np.array(
            [
                idxs
                for idxs in itertools.product(
                    *(range(len_dim) for len_dim in full.shape)
                )
            ]
        )
        return self._rng.choice(indexes_full, n_gridpts_to_mask, replace=False)


class MaskerIsland(Masker):
    def __init__(self, perc_to_keep: float):
        super().__init__()

        self._perc_to_keep = perc_to_keep

    def mask(self, full: torch.Tensor) -> torch.Tensor:
        lows, highs = self._range_idx_dim(full)
        res = torch.zeros_like(full)
        for idxs in itertools.product(*(range(len_dim) for len_dim in full.shape)):
            idxs_np = np.array(idxs)
            if np.all(idxs_np >= lows) and np.all(idxs_np < highs):
                res[idxs] = full[idxs]
        return res

    def _range_idx_dim(self, full: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        perc_mask_per_side = (1 - self._perc_to_keep) / 2

        lows, highs = [], []
        for size_dim in full.shape:
            lows.append(int(size_dim * perc_mask_per_side))
            highs.append(int(size_dim * (1 - perc_mask_per_side)))
        return np.array(lows), np.array(highs)


class Filter:
    def __init__(self, dataset: DatasetPde):
        self._dataset = dataset

    def filter(self, *ranges: tuple[float, float]) -> tuple[DatasetPde, DatasetPde]:
        self._check_ranges(*ranges)

        lhss_boundary, rhss_boundary = [], []
        lhss_internal, rhss_internal = [], []
        for lhs, rhs in self._dataset.dataset:
            if self._in_range(lhs, *ranges):
                if self._is_boundary(lhs, *ranges):
                    lhss_boundary.append(lhs)
                    rhss_boundary.append(rhs)
                else:
                    lhss_internal.append(lhs)
                    rhss_internal.append(rhs)

        return (
            DatasetPde.from_lhss_rhss_torch(lhss_boundary, rhss_boundary),
            DatasetPde.from_lhss_rhss_torch(lhss_internal, rhss_internal),
        )

    def _check_ranges(self, *ranges: tuple[float, float]) -> None:
        if self._dataset.n_dims != len(ranges):
            raise ValueError("number of ranges do NOT match number of dimensions")

    def _in_range(self, lhs: torch.Tensor, *ranges: tuple[float, float]) -> bool:
        return all((rg[0] <= val <= rg[1] for val, rg in zip(lhs, ranges)))

    def _is_boundary(
        self,
        lhs: torch.Tensor,
        *ranges: tuple[float, float],
    ) -> bool:
        for val, (pt_min, pt_max) in zip(lhs, ranges):
            if math.isclose(val, pt_min, abs_tol=0.0001) or math.isclose(
                val, pt_max, abs_tol=0.0001
            ):
                return True
        return False


class MultiEval:
    def __init__(
        self,
        eval_network: Callable[[torch.Tensor], torch.Tensor],
    ):
        self._eval_network = eval_network

    def loss_weighted(
        self, datasets: Iterable[DatasetPde], weights: Iterable[float]
    ) -> torch.Tensor:
        losses = torch.tensor([0.0])
        for weight, dataset in zip(weights, datasets):
            if not dataset.is_empty():
                losses += (
                    weight
                    * distance.Distance(
                        self._eval_network(dataset.lhss), dataset.rhss
                    ).mse()
                )

        return losses
