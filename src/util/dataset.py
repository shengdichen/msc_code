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
        if len(lhss.shape) != len(rhss.shape):
            raise ValueError("different number of dimensions of lhss and rhss")

    @property
    def lhss(self) -> torch.Tensor:
        return self._lhss

    @property
    def rhss(self) -> torch.Tensor:
        return self._rhss

    @property
    def lhss_rhss(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._lhss, self._rhss

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
    def from_torch(
        cls, lhss: list[torch.Tensor], rhss: list[torch.Tensor]
    ) -> "DatasetPde":
        if not lhss:
            lhss_torch, rhss_torch = torch.tensor([]), torch.tensor([])
        else:
            lhss_torch = torch.stack(lhss)
            rhss_torch = torch.stack(rhss)
        return cls(lhss_torch, rhss_torch)

    @classmethod
    def from_dataset(
        cls, *datasets: torch.utils.data.dataset.TensorDataset
    ) -> "DatasetPde":
        lhss: list[torch.Tensor] = []
        rhss: list[torch.Tensor] = []
        for dataset in datasets:
            for lhs, rhs in dataset:
                lhss.append(lhs)
                rhss.append(rhs)
        return cls.from_torch(lhss, rhss)

    @staticmethod
    def one_big_batch(dataset: torch.utils.data.dataset.TensorDataset) -> list:
        return list(torch.utils.data.DataLoader(dataset, batch_size=len(dataset)))[0]


class Masker:
    @abc.abstractmethod
    def mask(self, full: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def as_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def as_perc(self) -> float:
        raise NotImplementedError


class MaskerRandom(Masker):
    def __init__(
        self,
        perc_to_mask: float = 0.5,
        value_mask: float = 0.5,
        seed: typing.Optional[int] = None,
    ):
        super().__init__()

        self._rng = np.random.default_rng(seed=seed)
        self._perc_to_mask, self._value_mask = perc_to_mask, value_mask

        self._name = f"random_{self._perc_to_mask:.2}"

    def as_name(self) -> str:
        return self._name

    def as_perc(self) -> float:
        return self._perc_to_mask

    def mask(self, full: torch.Tensor) -> torch.Tensor:
        if math.isclose(self._perc_to_mask, 1.0):
            return (self._value_mask * torch.ones_like(full)).type_as(full)

        res = full.detach().clone()
        if math.isclose(self._perc_to_mask, 0.0):
            return res
        for idx in self._indexes_to_mask(full):
            res[tuple(idx)] = self._value_mask
        return res.type_as(full)

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
    def __init__(self, perc_to_keep: float, value_mask: float = 0.5):
        super().__init__()

        self._perc_to_keep, self._value_mask = perc_to_keep, value_mask
        self._name = f"island_{self._perc_to_keep:.2}"

    def as_name(self) -> str:
        return self._name

    def as_perc(self) -> float:
        return self._perc_to_keep

    def mask(self, full: torch.Tensor) -> torch.Tensor:
        if math.isclose(self._perc_to_keep, 1.0):
            return full.detach().clone()

        res = (self._value_mask * torch.ones_like(full)).type_as(full)
        if math.isclose(self._perc_to_keep, 0.0):
            return res

        lows, highs = self._range_idx_dim(full)
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
            DatasetPde.from_torch(lhss_boundary, rhss_boundary),
            DatasetPde.from_torch(lhss_internal, rhss_internal),
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


class Reorderer:
    @staticmethod
    def components_to_second(
        dataset: torch.utils.data.dataset.TensorDataset,
    ) -> torch.utils.data.dataset.TensorDataset:
        """
        lhss: [n_instances, x..., n_channels] -> [n_instances, n_channels, x...]
        rhss: [n_instances, x..., 1] -> [n_instances, 1, x...]
        """
        lhss, rhss = DatasetPde.from_dataset(dataset).lhss_rhss

        idxs = list(range(lhss.dim()))
        idxs.insert(1, idxs.pop())  # e.g. (0, 3, 1, 2) if 4-dimensional
        return torch.utils.data.TensorDataset(lhss.permute(idxs), rhss.permute(idxs))

    @staticmethod
    def components_to_last(
        dataset: torch.utils.data.dataset.TensorDataset,
    ) -> torch.utils.data.dataset.TensorDataset:
        """
        lhss: [n_instances, n_channels, x...] -> [n_instances, x..., n_channels]
        rhss: [n_instances, 1, x...] -> [n_instances, x..., 1]
        """
        lhss, rhss = DatasetPde.from_dataset(dataset).lhss_rhss

        idxs = list(range(lhss.dim()))
        idxs.append(idxs.pop(1))  # e.g. (0, 2, 3, 1) if 4-dimensional
        return torch.utils.data.TensorDataset(lhss.permute(idxs), rhss.permute(idxs))


class Normalizer:
    """
    lhss: [n_samples, n_channels_lhs, x...]
    rhss: [n_samples, n_channels_rhs, x...]
    """

    def __init__(
        self,
        min_lhss: torch.Tensor,
        max_lhss: torch.Tensor,
        shape_lhss: torch.Size,
        min_rhss: torch.Tensor,
        max_rhss: torch.Tensor,
        shape_rhss: torch.Size,
    ):
        # both: [1, n_channels_lhs, 1...]
        self._min_lhss, self._max_lhss = min_lhss, max_lhss
        self._shape_lhss = shape_lhss

        # both: [1, n_channels_rhs, 1...]
        self._min_rhss, self._max_rhss = min_rhss, max_rhss
        self._shape_rhss = shape_rhss

    @classmethod
    def from_dataset(
        cls, dataset: torch.utils.data.dataset.TensorDataset
    ) -> "Normalizer":
        return cls.from_lhss_rhss(*DatasetPde.from_dataset(dataset).lhss_rhss)

    @classmethod
    def from_lhss_rhss(cls, lhss: torch.Tensor, rhss: torch.Tensor) -> "Normalizer":
        return cls(*cls._minmax(lhss), *cls._minmax(rhss))

    @staticmethod
    def _minmax(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
        shape = target.shape

        target = Normalizer._swap_axes_sample_channel(target)  # channel to 1st axis
        target_flat = target.reshape(target.shape[0], -1)  # flatten each channel
        min_, max_ = target_flat.min(1).values, target_flat.max(1).values

        # expand to original dimension
        for __ in range(target.dim() - 1):
            min_ = min_.unsqueeze(-1)
            max_ = max_.unsqueeze(-1)

        # swap (min & max) back to 2nd axis
        min_, max_ = (
            Normalizer._swap_axes_sample_channel(min_),
            Normalizer._swap_axes_sample_channel(max_),
        )

        return min_, max_, shape

    @staticmethod
    def _swap_axes_sample_channel(target: torch.Tensor) -> torch.Tensor:
        # NOTE:
        #   (samples, channels) := (axis_1, axis_2) or (axis_2, axis_1)
        idxs = list(range(target.dim()))
        idxs[0], idxs[1] = idxs[1], idxs[0]
        return target.permute(idxs)

    def normalize_dataset(
        self, dataset: torch.utils.data.dataset.TensorDataset
    ) -> torch.utils.data.dataset.TensorDataset:
        lhss, rhss = DatasetPde.from_dataset(dataset).lhss_rhss
        return torch.utils.data.TensorDataset(
            self.normalize_lhss(lhss), self.normalize_rhss(rhss)
        )

    def normalize_lhss(self, lhss: torch.Tensor) -> torch.Tensor:
        self._check_shape(lhss, use_lhss=True)
        return self._normalize(lhss, self._min_lhss, self._max_lhss)

    def normalize_rhss(self, rhss: torch.Tensor) -> torch.Tensor:
        self._check_shape(rhss, use_lhss=False)
        return self._normalize(rhss, self._min_rhss, self._max_rhss)

    @staticmethod
    def _normalize(
        target: torch.Tensor, min_: torch.Tensor, max_: torch.Tensor
    ) -> torch.Tensor:
        return (target - min_) / (max_ - min_)

    def denormalize_lhss(self, lhss: torch.Tensor) -> torch.Tensor:
        self._check_shape(lhss, use_lhss=True)
        return self._denormalize(lhss, self._min_rhss, self._max_rhss)

    def denormalize_rhss(self, rhss: torch.Tensor) -> torch.Tensor:
        self._check_shape(rhss, use_lhss=False)
        return self._denormalize(rhss, self._min_rhss, self._max_rhss)

    def _denormalize(
        self, target: torch.Tensor, min_: torch.Tensor, max_: torch.Tensor
    ) -> torch.Tensor:
        return target * (max_ - min_) + min_

    def _check_shape(self, target: torch.Tensor, use_lhss: bool) -> None:
        if use_lhss:
            shape = self._shape_lhss
        else:
            shape = self._shape_rhss
        if target.shape[1:] != shape[1:]:
            raise ValueError(
                "norm> shape mismatch: "
                f"expects {shape}; but got {target.shape} instead"
            )


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
