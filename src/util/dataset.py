import abc
import itertools
import logging
import math
import random
import typing
from collections.abc import Callable, Iterable

import numpy as np
import torch

from src.definition import DEFINITION, T_DATASET
from src.numerics import distance, grid
from src.util import plot

logger = logging.getLogger(__name__)


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
    def dataset(self) -> T_DATASET:
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
    def from_dataset(cls, *datasets: T_DATASET) -> "DatasetPde":
        lhss: list[torch.Tensor] = []
        rhss: list[torch.Tensor] = []
        for dataset in datasets:
            for lhs, rhs in dataset:
                lhss.append(lhs)
                rhss.append(rhs)
        return cls.from_torch(lhss, rhss)

    @staticmethod
    def one_big_batch(dataset: T_DATASET) -> list:
        return list(torch.utils.data.DataLoader(dataset, batch_size=len(dataset)))[0]


class Splitter:
    def __init__(self, dataset_full: T_DATASET):
        self._dataset_full = dataset_full
        self._n_instances = len(self._dataset_full)

    def split(
        self, n_instances_eval: int, n_instances_train: int
    ) -> tuple[T_DATASET, T_DATASET]:
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


class Masker:
    def __init__(
        self,
        intensity: float = 0.5,
        intensity_spread: float = 0.1,
        value_mask: float = 0.5,
    ):
        self._intensity = intensity
        self._intensity_spread = intensity_spread
        self._intensity_min, self._intensity_max = (
            self._intensity - self._intensity_spread,
            self._intensity + self._intensity_spread,
        )

        self._value_mask = value_mask

        self._name: str
        self._name_human: str

    @abc.abstractmethod
    def mask(self, full: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    @property
    def name_human(self) -> str:
        return self._name_human

    @property
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, value: float) -> None:
        self._intensity = value

    def min_max(self, as_percentage: bool = False) -> tuple[float, float]:
        if as_percentage:
            return self._intensity_min * 100, self._intensity_max * 100
        return self._intensity_min, self._intensity_max

    def _sample_intensity(self) -> float:
        if math.isclose(self._intensity_spread, 0.0):
            return self._intensity

        intensity = random.uniform(
            self._intensity - self._intensity_spread,
            self._intensity + self._intensity_spread,
        )
        if intensity < 0.0:
            return 0.0
        if intensity > 1.0:
            return 1.0
        return intensity

    def _intensity_real(self, full: torch.Tensor, masked: torch.Tensor) -> float:
        return (
            np.count_nonzero(masked == self._value_mask) / np.prod(full.shape)
        ).item()

    def plot(self, resolution: int = 50) -> None:
        grids = grid.Grids(
            [
                grid.Grid.from_start_end(resolution, start=-1, end=+1),
                grid.Grid.from_start_end(resolution, start=-1, end=+1),
            ],
        )
        coords_x1, coords_x2 = grids.coords_as_mesh_torch()

        pt = plot.PlotIllustration(grids)
        values_mask = [0, 1 / 4, 1 / 2]
        pt.make_fig_ax(1 + len(values_mask))

        target = torch.cos(0.5 * torch.pi * coords_x1) * torch.cos(
            0.5 * torch.pi * coords_x2
        )

        _min, _max = 0.0, 1.0
        pt.plot_2d(
            pt.axs_2d[0],
            target,
            "cos × cos (unmasked)",  # no, this mult sign is not ascii
            _min=0.0,
            _max=+1.0,
            colormap="plasma",
        )
        pt.plot_3d(pt.axs_3d[0], target, _min=_min, _max=_max, colormap="plasma")

        targets = []
        titles = []
        for val in values_mask:
            self._value_mask = val
            targets.append(self.mask(target))
            titles.append(f"mask-value: {val:.2f}")
        pt.plot_targets_uniform(targets, titles, _min=_min, _max=_max, idx_ax_start=1)

        pt.finalize(
            DEFINITION.BIN_DIR / f"mask_{self._name}.png",
            title=f"Masking: {self.name_human}",
        )


class MaskerRandom(Masker):
    def __init__(
        self,
        intensity: float = 0.5,
        intensity_spread: float = 0.1,
        value_mask: float = 0.5,
        seed: typing.Optional[int] = 42,
    ):
        super().__init__(intensity, intensity_spread, value_mask)

        self._rng = np.random.default_rng(seed=seed)

        self._name = (
            f"random"
            "_"
            f"{(self._intensity - self._intensity_spread):.2}"
            "_"
            f"{(self._intensity + self._intensity_spread):.2}"
        )
        if not math.isclose(self._value_mask, 0.5):
            self._name = f"{self._name}_val_{self._value_mask:.1f}"
        if self._intensity_spread:
            self._name_human = (
                f"Random"
                " "
                f"[{self._intensity_min:.0%}"
                "-"
                f"{self._intensity_max:.0%}]"
            )
        else:
            self._name_human = f"Random {self._intensity:.0%}"

    @classmethod
    def from_min_max(
        cls,
        intensity_min: float = 0.4,
        intensity_max: float = 0.6,
        value_mask: float = 0.5,
        seed: typing.Optional[int] = None,
    ) -> "MaskerRandom":
        return cls(
            intensity=(intensity_max + intensity_min) / 2,
            intensity_spread=(intensity_max - intensity_min) / 2,
            value_mask=value_mask,
            seed=seed,
        )

    def mask(self, full: torch.Tensor) -> torch.Tensor:
        if math.isclose(self._intensity, 1.0):
            return (self._value_mask * torch.ones_like(full)).type_as(full)

        res = full.detach().clone()
        if math.isclose(self._intensity, 0.0):
            return res
        for idx in self._indexes_to_mask(full):
            res[tuple(idx)] = self._value_mask
        return res.type_as(full)

    def _indexes_to_mask(self, full: torch.Tensor) -> np.ndarray:
        n_gridpts_to_mask = int(self._sample_intensity() * np.prod(full.shape))

        indexes_full = np.array(
            list(itertools.product(*(range(len_dim) for len_dim in full.shape)))
        )
        return self._rng.choice(indexes_full, n_gridpts_to_mask, replace=False)


class MaskerIsland(Masker):
    def __init__(
        self, intensity: float, intensity_spread: float = 0.1, value_mask: float = 0.5
    ):
        super().__init__(intensity, intensity_spread, value_mask)

        self._name = (
            f"island"
            "_"
            f"{(self._intensity - self._intensity_spread):.2}"
            "_"
            f"{(self._intensity + self._intensity_spread):.2}"
        )
        if self._intensity_spread:
            self._name_human = (
                f"Island"
                " "
                f"[{self._intensity_min:.0%}"
                "-"
                f"{self._intensity_max:.0%}]"
            )
        else:
            self._name_human = f"Island {self._intensity:.0%}"

    @classmethod
    def from_min_max(
        cls,
        intensity_min: float = 0.4,
        intensity_max: float = 0.6,
        value_mask: float = 0.5,
    ) -> "MaskerIsland":
        return cls(
            intensity=(intensity_max + intensity_min) / 2,
            intensity_spread=(intensity_max - intensity_min) / 2,
            value_mask=value_mask,
        )

    def mask(self, full: torch.Tensor) -> torch.Tensor:
        if math.isclose(self._intensity, 0.0):
            return full.detach().clone()

        res = (self._value_mask * torch.ones_like(full)).type_as(full)
        if math.isclose(self._intensity, 1.0):
            return res

        lows, highs = self._range_idx_dim(full)
        for idxs in itertools.product(*(range(len_dim) for len_dim in full.shape)):
            idxs_np = np.array(idxs)
            if np.all(idxs_np >= lows) and np.all(idxs_np < highs):
                res[idxs] = full[idxs]
        logger.debug(
            "mask/island> intensity (expected, actual): "
            f"{self._intensity_real(full, res)}"
            ", "
            f"{self._intensity}"
        )
        return res

    def _range_idx_dim(self, full: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        perc_untouched_per_side = (1 - self._sample_intensity()) ** (1 / full.dim())
        perc_mask_per_side = (1 - perc_untouched_per_side) / 2

        lows, highs = [], []
        for size_dim in full.shape:
            lows.append(int(size_dim * perc_mask_per_side))
            highs.append(int(size_dim * (1 - perc_mask_per_side)))
        return np.array(lows), np.array(highs)


class MaskerRing(Masker):
    def __init__(
        self, intensity: float, intensity_spread: float = 0.1, value_mask: float = 0.5
    ):
        super().__init__(intensity, intensity_spread, value_mask)

        self._name = (
            f"ring"
            "_"
            f"{(self._intensity - self._intensity_spread):.2}"
            "_"
            f"{(self._intensity + self._intensity_spread):.2}"
        )
        if self._intensity_spread:
            self._name_human = (
                f"Ring"
                " "
                f"[{self._intensity_min:.0%}"
                "-"
                f"{self._intensity_max:.0%}]"
            )
        else:
            self._name_human = f"Ring {self._intensity:.0%}"

    @classmethod
    def from_min_max(
        cls,
        intensity_min: float = 0.4,
        intensity_max: float = 0.6,
        value_mask: float = 0.5,
    ) -> "MaskerRing":
        return cls(
            intensity=(intensity_max + intensity_min) / 2,
            intensity_spread=(intensity_max - intensity_min) / 2,
            value_mask=value_mask,
        )

    def mask(self, full: torch.Tensor) -> torch.Tensor:
        if math.isclose(self._intensity, 1.0):
            return (self._value_mask * torch.ones_like(full)).type_as(full)

        res = full.detach().clone()
        if math.isclose(self._intensity, 0.0):
            return res

        lows, highs = self._range_idx_dim(full)
        for idxs in itertools.product(*(range(len_dim) for len_dim in full.shape)):
            idxs_np = np.array(idxs)
            if np.all(idxs_np >= lows) and np.all(idxs_np < highs):
                res[idxs] = self._value_mask
        return res

    def _range_idx_dim(self, full: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        perc_untouched_per_side = self._sample_intensity() ** (1 / full.dim())
        perc_mask_per_side = (1 - perc_untouched_per_side) / 2

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
    def components_to_second_tensors(
        *tensors: torch.Tensor,
    ) -> typing.Sequence[torch.Tensor]:
        """
        tensor: [n_instances, x..., n_channels] -> [n_instances, n_channels, x...]
        """
        idxs = list(range(tensors[0].dim()))
        idxs.insert(1, idxs.pop())  # e.g. (0, 3, 1, 2) if 4-dimensional
        return [tensor.permute(idxs) for tensor in tensors]

    @staticmethod
    def components_to_second(dataset: T_DATASET) -> T_DATASET:
        """
        lhss: [n_instances, x..., n_channels_lhs] -> [n_instances, n_channels_lhs, x...]
        rhss: [n_instances, x..., n_channels_rhs] -> [n_instances, n_channels_rhs, x...]
        """
        lhss, rhss = DatasetPde.from_dataset(dataset).lhss_rhss
        return torch.utils.data.TensorDataset(
            *Reorderer.components_to_second_tensors(lhss, rhss)
        )

    @staticmethod
    def components_to_last_tensors(
        *tensors: torch.Tensor,
    ) -> typing.Sequence[torch.Tensor]:
        """
        tensor: [n_instances, n_channels, x...] -> [n_instances, x..., n_channels]
        """
        idxs = list(range(tensors[0].dim()))
        idxs.append(idxs.pop(1))  # e.g. (0, 2, 3, 1) if 4-dimensional
        return [tensor.permute(idxs) for tensor in tensors]

    @staticmethod
    def components_to_last(dataset: T_DATASET) -> T_DATASET:
        """
        lhss: [n_instances, n_channels_lhs, x...] -> [n_instances, x..., n_channels_lhs]
        rhss: [n_instances, n_channels_rhs, x...] -> [n_instances, x..., n_channels_rhs]
        """
        lhss, rhss = DatasetPde.from_dataset(dataset).lhss_rhss
        return torch.utils.data.TensorDataset(
            *Reorderer.components_to_last_tensors(lhss, rhss)
        )


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
    def from_dataset(cls, dataset: T_DATASET) -> "Normalizer":
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

    def normalize_dataset(self, dataset: T_DATASET) -> T_DATASET:
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
