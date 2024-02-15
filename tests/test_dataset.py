import torch

from src.numerics import grid
from src.numerics.equality import EqualityTorch
from src.util import dataset
from src.util.dataset import DatasetPde, Filter


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


class TestMask:
    def test_mask_random(self):
        full = torch.tensor(
            [
                [11, 12, 13, 14, 15],
                [21, 22, 23, 24, 25],
                [31, 32, 33, 34, 35],
                [41, 42, 43, 44, 45],
                [51, 52, 53, 54, 55],
                [61, 62, 63, 64, 65],
            ]
        )
        masker = dataset.MaskerRandom(full, 0.3, seed=42)
        assert torch.allclose(
            masker.mask(),
            torch.tensor(
                [
                    [11, 0, 0, 14, 15],
                    [21, 0, 23, 24, 25],
                    [0, 0, 33, 34, 35],
                    [0, 42, 0, 44, 45],
                    [0, 52, 53, 0, 55],
                    [61, 62, 63, 64, 65],
                ]
            ),
        )
        assert torch.allclose(
            masker.mask(),
            torch.tensor(
                [
                    [11, 12, 0, 14, 0],
                    [21, 22, 23, 24, 0],
                    [0, 32, 0, 34, 35],
                    [41, 42, 43, 44, 0],
                    [51, 52, 0, 54, 55],
                    [0, 62, 63, 64, 0],
                ]
            ),
        )


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
        ft = Filter(internal)

        boundary, internal = ft.filter((3.2, 3.5), (4.3, 4.6))
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
