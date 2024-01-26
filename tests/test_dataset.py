import torch

from src.pde.dataset import DatasetPde, Filter
from src.util import grid
from src.util.equality import EqualityTorch


class _Data:
    @staticmethod
    def make_dataset() -> torch.utils.data.dataset.TensorDataset:
        lhss = torch.tensor(
            [
                [1, 2],
                [1, 2],
                [3, 7],
                [7, 3],
            ],
            dtype=torch.float,
        )
        rhss = torch.tensor(
            [
                [10],
                [20],
                [30],
                [40],
            ],
            dtype=torch.float,
        )

        return torch.utils.data.TensorDataset(lhss, rhss)

    @staticmethod
    def make_dataset_large() -> torch.utils.data.dataset.TensorDataset:
        pass


class TestDatasetPde:
    def test_from_lhss_rhss_raw(self):
        lhss_raw = [
            [1, 2],
            [1, 2],
            [3, 7],
            [7, 3],
        ]
        rhss_raw = [10, 20, 30, 40]

        dataset_ours = DatasetPde.from_lhss_rhss_raw(lhss_raw, rhss_raw).dataset
        assert isinstance(dataset_ours, torch.utils.data.dataset.Dataset)

    def test_from_lhss_rhss_torch(self):
        # TODO: adapt this test to torch
        lhss_raw = [
            [1, 2],
            [1, 2],
            [3, 7],
            [7, 3],
        ]
        rhss_raw = [10, 20, 30, 40]

        dataset_ours = DatasetPde.from_lhss_rhss_raw(lhss_raw, rhss_raw).dataset
        assert isinstance(dataset_ours, torch.utils.data.dataset.Dataset)

    def test_from_datasets(self):
        d_pde = DatasetPde.from_datasets(_Data.make_dataset(), _Data.make_dataset())

        assert EqualityTorch(
            d_pde.lhss,
            torch.tensor(
                [
                    [1, 2],
                    [1, 2],
                    [3, 7],
                    [7, 3],
                    [1, 2],
                    [1, 2],
                    [3, 7],
                    [7, 3],
                ]
            ),
        ).is_close()
        assert EqualityTorch(
            d_pde.rhss,
            torch.tensor(
                [
                    [10],
                    [20],
                    [30],
                    [40],
                    [10],
                    [20],
                    [30],
                    [40],
                ]
            ),
        ).is_close()


class TestFilter:
    def _make_dataset(self) -> torch.utils.data.dataset.TensorDataset:
        gr1 = grid.Grid(10, stepsize=0.1, start=3.0)
        gr2 = grid.Grid(10, stepsize=0.1, start=4.0)
        grids = grid.Grids([gr1, gr2])

        lhss_boundary, rhss_boundary = [], []
        lhss_internal, rhss_internal = [], []

        for val_x1, val_x2 in grids.boundaries():
            lhss_boundary.append([val_x1, val_x2])
            rhss_boundary.append(42.0)

        for val_x1, val_x2 in grids.internals():
            lhss_internal.append([val_x1, val_x2])
            rhss_internal.append(10.0)

        return (
            DatasetPde.from_lhss_rhss_raw(lhss_boundary, rhss_boundary),
            DatasetPde.from_lhss_rhss_raw(lhss_internal, rhss_internal),
        )

    def test_filter(self):
        _, internal = self._make_dataset()
        filter = Filter(internal)

        boundary, internal = filter.filter((3.2, 3.5), (4.3, 4.6))
        assert EqualityTorch(
            boundary.lhss,
            torch.tensor(
                [
                    [3.2000, 4.3000],
                    [3.2000, 4.4000],
                    [3.2000, 4.5000],
                    [3.2000, 4.6000],
                    [3.3000, 4.3000],
                    [3.3000, 4.6000],
                    [3.4000, 4.3000],
                    [3.4000, 4.6000],
                    [3.5000, 4.3000],
                    [3.5000, 4.4000],
                    [3.5000, 4.5000],
                    [3.5000, 4.6000],
                ]
            ),
        ).is_close()
        assert EqualityTorch(
            boundary.rhss,
            torch.tensor(
                [
                    [10.0],
                    [10.0],
                    [10.0],
                    [10.0],
                    [10.0],
                    [10.0],
                    [10.0],
                    [10.0],
                    [10.0],
                    [10.0],
                    [10.0],
                    [10.0],
                ]
            ),
        ).is_close()

        assert EqualityTorch(
            internal.lhss,
            torch.tensor(
                [
                    [3.3000, 4.4000],
                    [3.3000, 4.5000],
                    [3.4000, 4.4000],
                    [3.4000, 4.5000],
                ],
            ),
        ).is_close()
        assert EqualityTorch(
            internal.rhss,
            torch.tensor(
                [
                    [10.0],
                    [10.0],
                    [10.0],
                    [10.0],
                ]
            ),
        ).is_close()
