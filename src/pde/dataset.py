import math

import torch


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
            lhss, rhss = torch.tensor([]), torch.tensor([])
        else:
            lhss, rhss = torch.stack(lhss), torch.stack(rhss).view(-1, 1)

        return cls(lhss, rhss)

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

    def one_big_batch(cls, dataset: torch.utils.data.dataset.TensorDataset) -> list:
        return list(torch.utils.data.DataLoader(dataset, batch_size=len(dataset)))[0]


class Filter:
    def __init__(self, dataset: DatasetPde):
        self._dataset = dataset

    def filter(self, *ranges: tuple[float, float]) -> tuple[DatasetPde, DatasetPde]:
        self._check_ranges(*ranges)

        lhss_boundary, rhss_boundary = [], []
        lhss_internal, rhss_internal = [], []
        for lhs, rhs in self._dataset.dataset:
            if self._in_range(lhs, *ranges):
                if self._is_boundary(lhs, rhs, *ranges):
                    lhss_boundary.append(lhs)
                    rhss_boundary.append(rhs)
                else:
                    lhss_internal.append(lhs)
                    rhss_internal.append(rhs)

        return (
            DatasetPde.from_lhss_rhss_torch(lhss_boundary, rhss_boundary),
            DatasetPde.from_lhss_rhss_torch(lhss_internal, rhss_internal),
        )

    def _convert_to_torch(
        self, lhss: list[torch.Tensor], rhss: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not lhss:
            return torch.tensor([]), torch.tensor([])
        return torch.stack(lhss), torch.stack(rhss).view(-1, 1)

    def _check_ranges(self, *ranges: tuple[float, float]) -> None:
        if self._dataset.n_dims != len(ranges):
            raise ValueError("number of ranges do NOT match number of dimensions")

    def _in_range(self, lhs: torch.Tensor, *ranges: tuple[float, float]) -> bool:
        return all((rg[0] <= val <= rg[1] for val, rg in zip(lhs, ranges)))

    def _is_boundary(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *ranges: tuple[float, float],
    ) -> bool:
        for val, (min, max) in zip(lhs, ranges):
            if math.isclose(val, min, abs_tol=0.0001) or math.isclose(
                val, max, abs_tol=0.0001
            ):
                return True
        return False
