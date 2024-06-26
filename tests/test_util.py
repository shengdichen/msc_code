import random

import torch

from src.numerics import equality, grid
from src.pde.poisson import dataset as poisson_ds
from src.pde.wave import dataset as wave_ds
from src.util import dataset


class TestMask:
    def test_mask_random_extremes(self):
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
        masker = dataset.MaskerRandom(0.0)
        assert torch.allclose(
            masker.mask(full),
            torch.tensor(
                [
                    [11, 12, 13, 14, 15],
                    [21, 22, 23, 24, 25],
                    [31, 32, 33, 34, 35],
                    [41, 42, 43, 44, 45],
                    [51, 52, 53, 54, 55],
                    [61, 62, 63, 64, 65],
                ]
            ),
        )

        masker = dataset.MaskerRandom(1.0)
        assert torch.allclose(masker.mask(full), torch.zeros_like(full))

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
        masker = dataset.MaskerRandom(0.3, intensity_spread=0.0, value_mask=0, seed=42)

        assert torch.allclose(
            masker.mask(full),
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
        # repeated drawing yields different result
        assert torch.allclose(
            masker.mask(full),
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

    def test_mask_random_custom_value(self):
        full = torch.tensor(
            [
                [11, 12, 13, 14, 15],
                [21, 22, 23, 24, 25],
                [31, 32, 33, 34, 35],
                [41, 42, 43, 44, 45],
                [51, 52, 53, 54, 55],
                [61, 62, 63, 64, 65],
            ]
        ).type(torch.float)
        masker = dataset.MaskerRandom(
            0.3, intensity_spread=0.0, value_mask=0.5, seed=42
        )
        assert torch.allclose(
            masker.mask(full),
            torch.tensor(
                [
                    [11, 0.5, 0.5, 14, 15],
                    [21, 0.5, 23, 24, 25],
                    [0.5, 0.5, 33, 34, 35],
                    [0.5, 42, 0.5, 44, 45],
                    [0.5, 52, 53, 0.5, 55],
                    [61, 62, 63, 64, 65],
                ]
            ),
        )

    def test_mask_random_with_spread(self):
        value_mask = 7.3
        full = torch.rand((20, 20)).type(torch.float)
        masker = dataset.MaskerRandom(
            0.5, intensity_spread=0.1, value_mask=value_mask, seed=42
        )
        random.seed(42)

        for n_values_masked in [211, 162, 182, 177, 218]:
            assert (
                torch.count_nonzero(masker.mask(full) == value_mask) == n_values_masked
            )

    def test_mask_island_with_spread(self):
        value_mask = 7.3
        full = torch.rand((40, 40)).type(torch.float)
        masker = dataset.MaskerIsland(0.5, intensity_spread=0.1, value_mask=value_mask)
        random.seed(42)

        for n_values_masked in [1239, 1071, 1159, 1071, 1239]:
            assert (
                torch.count_nonzero(masker.mask(full) == value_mask) == n_values_masked
            )

    def test_mask_island_extremes(self) -> None:
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
        masker = dataset.MaskerIsland(0.0)
        assert torch.allclose(masker.mask(full), full)

        masker = dataset.MaskerIsland(1.0)
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
        assert torch.allclose(masker.mask(full), torch.zeros_like(full))

    def test_mask_island(self) -> None:
        full = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24, 25, 27, 27, 28, 29],
                [30, 31, 32, 33, 34, 35, 38, 37, 38, 39],
                [40, 41, 42, 43, 44, 45, 49, 47, 48, 49],
                [50, 51, 52, 53, 54, 55, 50, 57, 58, 59],
                [60, 61, 62, 63, 64, 65, 61, 67, 68, 69],
                [70, 71, 72, 73, 74, 75, 72, 77, 78, 79],
                [80, 81, 82, 83, 84, 85, 83, 87, 88, 89],
                [90, 91, 92, 93, 94, 95, 94, 97, 98, 99],
            ]
        )
        masker = dataset.MaskerIsland(0.5, intensity_spread=0.0, value_mask=0)
        for __ in range(2):  # repeated drawing yields the same result
            assert torch.allclose(
                masker.mask(full),
                torch.tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 32, 33, 34, 35, 38, 0, 0, 0],
                        [0, 0, 42, 43, 44, 45, 49, 0, 0, 0],
                        [0, 0, 52, 53, 54, 55, 50, 0, 0, 0],
                        [0, 0, 62, 63, 64, 65, 61, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
            )

    def test_mask_custom_value(self) -> None:
        full = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24, 25, 27, 27, 28, 29],
                [30, 31, 32, 33, 34, 35, 38, 37, 38, 39],
                [40, 41, 42, 43, 44, 45, 49, 47, 48, 49],
                [50, 51, 52, 53, 54, 55, 50, 57, 58, 59],
                [60, 61, 62, 63, 64, 65, 61, 67, 68, 69],
                [70, 71, 72, 73, 74, 75, 72, 77, 78, 79],
                [80, 81, 82, 83, 84, 85, 83, 87, 88, 89],
                [90, 91, 92, 93, 94, 95, 94, 97, 98, 99],
            ]
        )
        masker = dataset.MaskerIsland(0.5, intensity_spread=0.0, value_mask=3)
        for __ in range(2):  # repeated drawing yields the same result
            assert torch.allclose(
                masker.mask(full),
                torch.tensor(
                    [
                        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                        [3, 3, 32, 33, 34, 35, 38, 3, 3, 3],
                        [3, 3, 42, 43, 44, 45, 49, 3, 3, 3],
                        [3, 3, 52, 53, 54, 55, 50, 3, 3, 3],
                        [3, 3, 62, 63, 64, 65, 61, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
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
            rhss_boundary.append([42.0])

        for val_x1, val_x2 in grids.internals():
            lhss_internal.append([val_x1, val_x2])
            rhss_internal.append([10.0])

        return (
            dataset.DatasetPde(
                torch.tensor(lhss_boundary), torch.tensor(rhss_boundary)
            ),
            dataset.DatasetPde(
                torch.tensor(lhss_internal), torch.tensor(rhss_internal)
            ),
        )

    def test_filter(self):
        _, internal = self._make_dataset()
        ft = dataset.Filter(internal)

        boundary, internal = ft.filter((3.2, 3.5), (4.3, 4.6))
        assert equality.EqualityTorch(
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
        assert equality.EqualityTorch(
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

        assert equality.EqualityTorch(
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
        assert equality.EqualityTorch(
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


class TestReordering:
    def _make_ds(
        self, size_x1: int = 10, size_x2: int = 15, n_instances: int = 5
    ) -> torch.utils.data.dataset.TensorDataset:
        gr = grid.Grids(
            [
                grid.Grid(size_x1, stepsize=0.1, start=3.0),
                grid.Grid(size_x2, stepsize=0.1, start=4.0),
            ]
        )
        ds = wave_ds.DatasetMaskedSingleWave(
            poisson_ds.DatasetSin(gr), dataset.MaskerIsland(0.5)
        )
        ds.as_train(n_instances)
        return ds.dataset_masked

    def test_reordering(self) -> None:
        size_x1, size_x2, n_instances = 10, 15, 5
        ds = self._make_ds(size_x1, size_x2, n_instances)

        for lhs, rhs in ds:
            assert lhs.shape == (8, size_x1, size_x2)
            assert rhs.shape == (1, size_x1, size_x2)

        ds = dataset.Reorderer.components_to_last(ds)
        for lhs, rhs in ds:
            assert lhs.shape == (size_x1, size_x2, 8)
            assert rhs.shape == (size_x1, size_x2, 1)

        ds = dataset.Reorderer.components_to_second(ds)
        for lhs, rhs in ds:
            assert lhs.shape == (8, size_x1, size_x2)
            assert rhs.shape == (1, size_x1, size_x2)


class TestNormalization:
    def _make_ds(
        self, size_x1: int = 5, size_x2: int = 5, n_instances: int = 5
    ) -> torch.utils.data.dataset.TensorDataset:
        gr = grid.Grids(
            [
                grid.Grid(size_x1, stepsize=0.1, start=3.0),
                grid.Grid(size_x2, stepsize=0.1, start=4.0),
            ]
        )
        ds = wave_ds.DatasetMaskedSingleWave(
            poisson_ds.DatasetSin(gr), dataset.MaskerIsland(0.5)
        )
        ds.as_train(n_instances)
        return ds.dataset_masked

    def _make_grid_normalized(self) -> tuple[torch.Tensor, torch.Tensor]:
        grid_x1 = grid.Grid(5, stepsize=0.25, start=0.0)
        grid_x2 = grid.Grid(5, stepsize=0.25, start=0.0)
        coords_x1, coords_x2 = grid.Grids([grid_x1, grid_x2]).coords_as_mesh_torch()
        return coords_x1, coords_x2

    def test_normalize_dataset_grid(self) -> None:
        coords_x1, coords_x2 = grid.Grids(
            [
                grid.Grid(5, stepsize=0.25, start=10.0),
                grid.Grid(5, stepsize=0.5, start=20.0),
            ]
        ).coords_as_mesh_torch()
        ds = torch.utils.data.TensorDataset(
            torch.stack([coords_x1, coords_x2]).unsqueeze(0),  # 1, 2, x1, x2
            torch.stack([coords_x1, coords_x2]).unsqueeze(0),  # 1, 2, x1, x2
        )
        coords_x1_ref, coords_x2_ref = self._make_grid_normalized()

        normalizer = dataset.Normalizer.from_dataset(ds)
        ds_norm = normalizer.normalize_dataset(ds)
        lhss_norm, rhss_norm = dataset.DatasetPde.from_dataset(ds_norm).lhss_rhss
        assert equality.EqualityTorch(lhss_norm[0, 0], coords_x1_ref).is_equal()
        assert equality.EqualityTorch(lhss_norm[0, 1], coords_x2_ref).is_equal()
        assert equality.EqualityTorch(rhss_norm[0, 0], coords_x1_ref).is_equal()
        assert equality.EqualityTorch(rhss_norm[0, 1], coords_x2_ref).is_equal()

        lhss_denorm, rhss_denorm = (
            normalizer.denormalize_rhss(lhss_norm),
            normalizer.denormalize_rhss(rhss_norm),
        )
        assert equality.EqualityTorch(lhss_denorm[0, 0], coords_x1).is_equal()
        assert equality.EqualityTorch(lhss_denorm[0, 1], coords_x2).is_equal()
        assert equality.EqualityTorch(rhss_denorm[0, 0], coords_x1).is_equal()
        assert equality.EqualityTorch(rhss_denorm[0, 1], coords_x2).is_equal()

    def test_normalize_dataset(self) -> None:
        ds = self._make_ds()
        ds = dataset.Normalizer.from_dataset(ds).normalize_dataset(ds)
        coords_x1, coords_x2 = self._make_grid_normalized()

        for lhs, rhs in ds:
            assert rhs.min() >= 0.0
            assert rhs.max() <= 1.0
            for lhs_component in lhs:
                assert lhs_component.min() >= 0.0
                assert lhs_component.max() <= 1.0

            assert equality.EqualityTorch(lhs[2], coords_x1)
            assert equality.EqualityTorch(lhs[3], coords_x2)
