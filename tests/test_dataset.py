import math

import numpy as np
import torch

from src.numerics import equality, grid
from src.pde.heat import dataset as heat_ds
from src.pde.poisson import dataset as poisson_ds
from src.pde.wave import dataset as wave_ds
from src.util import dataset


class TestDatasetPoisson:
    def test_raw(self):
        torch.manual_seed(42)
        ds = poisson_ds.DatasetSin(
            grid.Grids(
                [
                    grid.Grid.from_start_end(5, start=-1.0, end=1.0),
                    grid.Grid.from_start_end(5, start=-1.0, end=1.0),
                ],
            ),
        )

        solution, source = ds.solve_instance()
        assert equality.EqualityTorch(
            solution,
            torch.tensor(
                [
                    [-2.2792e-16, 7.6125e-09, 0.0000e00, -7.6125e-09, 2.2792e-16],
                    [3.2879e-10, 1.5801e-02, 0.0000e00, -1.5801e-02, -3.2879e-10],
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                    [-3.2879e-10, -1.5801e-02, 0.0000e00, 1.5801e-02, 3.2879e-10],
                    [2.2792e-16, -7.6125e-09, 0.0000e00, 7.6125e-09, -2.2792e-16],
                ],
            ),
        ).is_close()
        assert equality.EqualityTorch(
            source,
            torch.tensor(
                [
                    [-6.7459e-14, 9.7628e-07, 0.0000e00, -9.7628e-07, 6.7459e-14],
                    [1.3757e-07, 1.3621e00, 0.0000e00, -1.3621e00, -1.3757e-07],
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                    [-1.3757e-07, -1.3621e00, 0.0000e00, 1.3621e00, 1.3757e-07],
                    [6.7459e-14, -9.7628e-07, 0.0000e00, 9.7628e-07, -6.7459e-14],
                ],
            ),
        ).is_close()

        solution, source = ds.solve_instance()
        assert equality.EqualityTorch(
            solution,
            torch.tensor(
                [
                    [-1.5121e-15, 3.1254e-09, 0.0000e00, -3.1254e-09, 1.5121e-15],
                    [-5.3496e-09, 2.7873e-02, 0.0000e00, -2.7873e-02, 5.3496e-09],
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                    [5.3496e-09, -2.7873e-02, 0.0000e00, 2.7873e-02, -5.3496e-09],
                    [1.5121e-15, -3.1254e-09, 0.0000e00, 3.1254e-09, -1.5121e-15],
                ],
            ),
        ).is_close()
        assert equality.EqualityTorch(
            source,
            torch.tensor(
                [
                    [-4.6477e-13, 8.7161e-07, 0.0000e00, -8.7161e-07, 4.6477e-13],
                    [-8.9311e-07, 1.3721e00, 0.0000e00, -1.3721e00, 8.9311e-07],
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                    [8.9311e-07, -1.3721e00, 0.0000e00, 1.3721e00, -8.9311e-07],
                    [4.6477e-13, -8.7161e-07, 0.0000e00, 8.7161e-07, -4.6477e-13],
                ],
            ),
        ).is_close()


class TestDatasetHeat:
    def test_raw(self):
        torch.manual_seed(42)
        ds = heat_ds.DatasetHeat(
            grid.Grids(
                [
                    grid.Grid.from_start_end(5, start=-1.0, end=1.0),
                    grid.Grid.from_start_end(5, start=-1.0, end=1.0),
                ],
            ),
            grid.GridTime.from_start_end(n_pts=100, start=0.0, end=0.01),
        )

        u_start, u_end = ds.solve_instance()
        assert equality.EqualityTorch(
            u_start,
            torch.tensor(
                [
                    [1.4342e-13, -7.2425e-08, 0.0000e00, 7.2425e-08, -1.4342e-13],
                    [-7.2425e-08, 5.3027e-01, 0.0000e00, -5.3027e-01, 7.2425e-08],
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                    [7.2425e-08, -5.3027e-01, 0.0000e00, 5.3027e-01, -7.2425e-08],
                    [-1.4342e-13, 7.2425e-08, 0.0000e00, -7.2425e-08, 1.4342e-13],
                ],
            ),
        ).is_close()
        assert equality.EqualityTorch(
            u_end,
            torch.tensor(
                [
                    [2.0818e-14, -5.5685e-08, 0.0000e00, 5.5685e-08, -2.0818e-14],
                    [-5.5685e-08, 5.8740e-01, 0.0000e00, -5.8740e-01, 5.5685e-08],
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                    [5.5685e-08, -5.8740e-01, 0.0000e00, 5.8740e-01, -5.5685e-08],
                    [-2.0818e-14, 5.5685e-08, 0.0000e00, -5.5685e-08, 2.0818e-14],
                ],
            ),
        ).is_close()

        u_start, u_end = ds.solve_instance()
        assert equality.EqualityTorch(
            u_start,
            torch.tensor(
                [
                    [7.6033e-14, 7.5433e-09, 0.0000e00, -7.5433e-09, -7.6033e-14],
                    [7.5433e-09, -7.0596e-01, 0.0000e00, 7.0596e-01, -7.5433e-09],
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                    [-7.5433e-09, 7.0596e-01, 0.0000e00, -7.0596e-01, 7.5433e-09],
                    [-7.6033e-14, -7.5433e-09, 0.0000e00, 7.5433e-09, 7.6033e-14],
                ],
            ),
        ).is_close()
        assert equality.EqualityTorch(
            u_end,
            torch.tensor(
                [
                    [4.3171e-15, 1.3762e-08, 0.0000e00, -1.3762e-08, -4.3171e-15],
                    [1.3762e-08, -2.6042e-01, 0.0000e00, 2.6042e-01, -1.3762e-08],
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                    [-1.3762e-08, 2.6042e-01, 0.0000e00, -2.6042e-01, 1.3762e-08],
                    [-4.3171e-15, -1.3762e-08, 0.0000e00, 1.3762e-08, 4.3171e-15],
                ],
            ),
        ).is_close()


class TestDatasetWave:
    def test_raw(self):
        torch.manual_seed(42)
        ds = wave_ds.DatasetWave(
            grid.Grids(
                [
                    grid.Grid.from_start_end(5, start=0.0, end=1.0),
                    grid.Grid.from_start_end(5, start=0.0, end=1.0),
                ],
            ),
            grid.GridTime(n_pts=100, stepsize=0.1),
        )

        u_start, u_end = ds.solve_instance()
        assert equality.EqualityTorch(
            u_start,
            torch.tensor(
                [
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                    [0.0000e00, 3.4195e-01, 3.5793e-01, 9.1426e-02, 3.3005e-08],
                    [0.0000e00, 4.3004e-01, 4.0722e-01, -1.5845e-01, 5.2478e-08],
                    [0.0000e00, 4.2974e-01, 3.5946e-01, 3.1705e-01, -2.3610e-08],
                    [0.0000e00, -2.3826e-08, -2.8986e-08, -1.0303e-08, 3.6038e-16],
                ],
            ),
        ).is_close()
        assert equality.EqualityTorch(
            u_end,
            torch.tensor(
                [
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                    [0.0000e00, -4.9653e-02, -2.3845e-01, -1.7271e-01, 4.4420e-08],
                    [0.0000e00, 1.6214e-01, 2.0750e-02, -2.0350e-01, 6.3943e-08],
                    [0.0000e00, 9.2996e-02, -1.9201e-01, -1.4835e-01, 4.3900e-08],
                    [0.0000e00, -1.1512e-08, 4.4674e-08, 2.4138e-08, -7.4556e-15],
                ],
            ),
        ).is_close()

        u_start, u_end = ds.solve_instance()
        assert equality.EqualityTorch(
            u_start,
            torch.tensor(
                [
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                    [0.0000e00, -3.2649e-02, 2.1591e-01, 2.7863e-02, 2.2469e-09],
                    [0.0000e00, 4.4148e-01, 5.8092e-01, 2.6200e-01, -8.8006e-09],
                    [0.0000e00, 2.3822e-01, 3.5872e-01, 2.1863e-01, -4.9809e-08],
                    [0.0000e00, -5.6290e-08, -7.1059e-08, 8.9396e-09, -4.2379e-15],
                ],
            ),
        ).is_close()
        assert equality.EqualityTorch(
            u_end,
            torch.tensor(
                [
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                    [0.0000e00, -3.5165e-02, -1.1158e-01, -8.4793e-02, 2.9084e-08],
                    [0.0000e00, -5.1548e-02, -2.8626e-01, -1.4097e-01, 1.8288e-08],
                    [0.0000e00, 7.3926e-02, 3.1532e-02, 9.3042e-02, 4.0589e-09],
                    [0.0000e00, -2.4321e-08, 2.2960e-08, -2.1556e-08, -2.4459e-15],
                ],
            ),
        ).is_close()


class TestDatasetMasked:
    def _size_grid(self) -> int:
        return 5

    def _value_mask(self) -> float:
        return 7.3  # deliberately NOT within [0, 1] range

    def _grid(self) -> grid.Grids:
        return grid.Grids(
            [
                grid.Grid(self._size_grid(), stepsize=0.1, start=3.0),
                grid.Grid(self._size_grid(), stepsize=0.1, start=4.0),
            ],
        )

    def test_mask_single_wave(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)

        grids = grid.Grids(
            [
                grid.Grid.from_start_end(6, start=0.0, end=1.0),
                grid.Grid.from_start_end(6, start=0.0, end=1.0),
            ],
        )
        grid_time = grid.GridTime.from_start_end_only(end=10.0)
        mask = dataset.MaskerRandom(intensity=0.2, value_mask=7, seed=42)

        train = wave_ds.DatasetMaskedSingleWave(
            wave_ds.DatasetWave(grids, grid_time), mask
        )
        train.as_train(1)

        lhs_0_truth = torch.tensor(
            [
                [0.0713, 0.0713, 0.0713, 0.0713, 0.0713, 0.0713],
                [0.0713, 0.7953, 0.6472, 0.6603, 0.2672, 0.0713],
                [0.0713, 0.5957, 1.0000, 0.6094, 0.0000, 0.0713],
                [0.0713, 0.4031, 0.7003, 0.9699, 0.5060, 0.0713],
                [0.0713, 0.1757, 0.1750, 0.7983, 0.3518, 0.0713],
                [0.0713, 0.0713, 0.0713, 0.0713, 0.0713, 0.0713],
            ]
        )
        coords_x1_truth = torch.tensor(
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
                [0.4000, 0.4000, 0.4000, 0.4000, 0.4000, 0.4000],
                [0.6000, 0.6000, 0.6000, 0.6000, 0.6000, 0.6000],
                [0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000],
                [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            ]
        )
        coords_x2_truth = torch.tensor(
            [
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
            ]
        )
        for instance in train._dataset_unmasked:  # pylint: disable=protected-access
            assert equality.EqualityTorch(instance[0][0], lhs_0_truth).is_close()
            assert equality.EqualityTorch(
                instance[0][train.MASK_IDX],
                torch.tensor(
                    [
                        [0.7049, 0.7049, 0.7049, 0.7049, 0.7049, 0.7049],
                        [0.7049, 0.4287, 0.4630, 0.6574, 1.0000, 0.7049],
                        [0.7049, 0.7162, 0.1077, 0.5666, 0.6369, 0.7049],
                        [0.7049, 0.9527, 0.5242, 0.6440, 0.2081, 0.7049],
                        [0.7049, 0.8911, 0.4814, 0.0000, 0.3809, 0.7049],
                        [0.7049, 0.7049, 0.7049, 0.7049, 0.7049, 0.7049],
                    ]
                ),
            ).is_close()
            assert equality.EqualityTorch(instance[0][2], coords_x1_truth).is_close()
            assert equality.EqualityTorch(instance[0][3], coords_x2_truth).is_close()

        for instance in train.dataset_masked:
            assert equality.EqualityTorch(instance[0][0], lhs_0_truth).is_close()
            assert equality.EqualityTorch(
                instance[0][train.MASK_IDX],
                torch.tensor(
                    [
                        [0.7049, 7.0000, 7.0000, 0.7049, 0.7049, 0.7049],
                        [7.0000, 0.4287, 0.4630, 0.6574, 1.0000, 0.7049],
                        [0.7049, 0.7162, 0.1077, 7.0000, 0.6369, 0.7049],
                        [0.7049, 0.9527, 7.0000, 0.6440, 0.2081, 0.7049],
                        [0.7049, 0.8911, 0.4814, 0.0000, 0.3809, 0.7049],
                        [0.7049, 7.0000, 0.7049, 0.7049, 7.0000, 0.7049],
                    ]
                ),
            ).is_close()
            assert equality.EqualityTorch(instance[0][2], coords_x1_truth).is_close()
            assert equality.EqualityTorch(instance[0][3], coords_x2_truth).is_close()

        train.remask()
        for instance in train.dataset_masked:
            assert equality.EqualityTorch(instance[0][0], lhs_0_truth).is_close()
            assert equality.EqualityTorch(
                instance[0][train.MASK_IDX],
                torch.tensor(
                    [
                        [0.7049, 0.7049, 7.0000, 7.0000, 0.7049, 0.7049],
                        [0.7049, 0.4287, 0.4630, 0.6574, 1.0000, 0.7049],
                        [0.7049, 0.7162, 7.0000, 0.5666, 0.6369, 0.7049],
                        [0.7049, 0.9527, 7.0000, 0.6440, 0.2081, 7.0000],
                        [0.7049, 7.0000, 0.4814, 0.0000, 0.3809, 7.0000],
                        [0.7049, 0.7049, 7.0000, 0.7049, 0.7049, 0.7049],
                    ]
                ),
            ).is_close()
            assert equality.EqualityTorch(instance[0][2], coords_x1_truth).is_close()
            assert equality.EqualityTorch(instance[0][3], coords_x2_truth).is_close()

    def test_mask_double(self) -> None:
        torch.manual_seed(42)
        gr = self._grid()

        ds = poisson_ds.DatasetPoissonMaskedSolutionSource(gr).make(
            poisson_ds.DatasetSin(gr, constant_multiplier=200).as_dataset(
                n_instances=2
            ),
            dataset.MaskerRandom(
                0.5, intensity_spread=0.0, value_mask=self._value_mask()
            ),
            dataset.MaskerRandom(
                0.5, intensity_spread=0.0, value_mask=self._value_mask()
            ),
        )

        for lhs, __ in ds:
            assert torch.count_nonzero(lhs[6] == self._value_mask()) >= math.floor(
                self._size_grid() ** 2 / 2
            )  # solution, masked
            assert torch.count_nonzero(lhs[7] == self._value_mask()) >= math.floor(
                self._size_grid() ** 2 / 2
            )  # source, masked
            break
