import torch

from src.definition import DEFINITION
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
    def _setup(self) -> tuple[wave_ds.DatasetWave, dataset.MaskerRandom]:
        DEFINITION.seed()

        grids = grid.Grids(
            [
                grid.Grid.from_start_end(6, start=0.0, end=1.0),
                grid.Grid.from_start_end(6, start=0.0, end=1.0),
            ],
        )
        grid_time = grid.GridTime.from_start_end_only(end=10.0)
        mask = dataset.MaskerRandom(intensity=0.2, intensity_spread=0.0, value_mask=7)

        return wave_ds.DatasetWave(grids, grid_time), mask

    def test_mask_single_wave(self) -> None:
        ds = wave_ds.DatasetMaskedSingleWave(*self._setup())
        ds.as_train(1)

        lhs_0_truth = torch.tensor(
            [
                [0.3683, 0.3683, 0.3683, 0.3683, 0.3683, 0.3683],
                [0.3683, 0.8110, 0.7109, 0.8207, 0.4728, 0.3683],
                [0.3683, 0.7537, 0.8019, 0.6743, 0.0000, 0.3683],
                [0.3683, 0.9627, 1.0000, 0.7434, 0.3539, 0.3683],
                [0.3683, 0.7628, 0.7301, 0.7690, 0.7184, 0.3683],
                [0.3683, 0.3683, 0.3683, 0.3683, 0.3683, 0.3683],
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
        for instance in ds._dataset_unmasked:  # pylint: disable=protected-access
            assert equality.EqualityTorch(instance[0][0], lhs_0_truth).is_close()
            assert equality.EqualityTorch(
                instance[0][1],  # u_T
                torch.tensor(
                    [
                        [0.5333, 0.5333, 0.5333, 0.5333, 0.5333, 0.5333],
                        [0.5333, 0.5052, 0.1167, 0.1636, 0.2603, 0.5333],
                        [0.5333, 0.8098, 0.2663, 0.2481, 0.0000, 0.5333],
                        [0.5333, 1.0000, 0.7210, 0.5643, 0.0701, 0.5333],
                        [0.5333, 0.7648, 0.1161, 0.0098, 0.3040, 0.5333],
                        [0.5333, 0.5333, 0.5333, 0.5333, 0.5333, 0.5333],
                    ]
                ),
            ).is_close()
            assert equality.EqualityTorch(instance[0][2], coords_x1_truth).is_close()
            assert equality.EqualityTorch(instance[0][3], coords_x2_truth).is_close()

        for instance in ds.dataset_masked:
            assert equality.EqualityTorch(instance[0][0], lhs_0_truth).is_close()
            assert equality.EqualityTorch(
                instance[0][1],  # u_T
                torch.tensor(
                    [
                        [0.5333, 0.5333, 7.0000, 7.0000, 0.5333, 0.5333],
                        [0.5333, 0.5052, 0.1167, 0.1636, 0.2603, 0.5333],
                        [0.5333, 0.8098, 7.0000, 0.2481, 0.0000, 0.5333],
                        [0.5333, 1.0000, 7.0000, 0.5643, 0.0701, 7.0000],
                        [0.5333, 0.7648, 0.1161, 0.0098, 0.3040, 0.5333],
                        [7.0000, 0.5333, 0.5333, 7.0000, 0.5333, 0.5333],
                    ]
                ),
            ).is_close()
            assert equality.EqualityTorch(instance[0][2], coords_x1_truth).is_close()
            assert equality.EqualityTorch(instance[0][3], coords_x2_truth).is_close()

        ds.remask()
        for instance in ds.dataset_masked:
            assert equality.EqualityTorch(instance[0][0], lhs_0_truth).is_close()
            assert equality.EqualityTorch(
                instance[0][1],
                torch.tensor(
                    [
                        [0.5333, 0.5333, 0.5333, 0.5333, 7.0000, 0.5333],
                        [0.5333, 0.5052, 0.1167, 0.1636, 0.2603, 0.5333],
                        [0.5333, 0.8098, 0.2663, 0.2481, 7.0000, 0.5333],
                        [0.5333, 1.0000, 0.7210, 0.5643, 7.0000, 0.5333],
                        [0.5333, 7.0000, 0.1161, 0.0098, 0.3040, 7.0000],
                        [7.0000, 0.5333, 0.5333, 0.5333, 0.5333, 7.0000],
                    ]
                ),
            ).is_close()
            assert equality.EqualityTorch(instance[0][2], coords_x1_truth).is_close()
            assert equality.EqualityTorch(instance[0][3], coords_x2_truth).is_close()

    def test_mask_double_wave(self):
        ds = wave_ds.DatasetMaskedDoubleWave(*self._setup())
        ds.as_train(1)

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
        for instance in ds._dataset_unmasked:  # pylint: disable=protected-access
            assert equality.EqualityTorch(
                instance[0][0],
                torch.tensor(
                    [
                        [0.3683, 0.3683, 0.3683, 0.3683, 0.3683, 0.3683],
                        [0.3683, 0.8110, 0.7109, 0.8207, 0.4728, 0.3683],
                        [0.3683, 0.7537, 0.8019, 0.6743, 0.0000, 0.3683],
                        [0.3683, 0.9627, 1.0000, 0.7434, 0.3539, 0.3683],
                        [0.3683, 0.7628, 0.7301, 0.7690, 0.7184, 0.3683],
                        [0.3683, 0.3683, 0.3683, 0.3683, 0.3683, 0.3683],
                    ]
                ),
            ).is_close()
            assert equality.EqualityTorch(
                instance[0][1],
                torch.tensor(
                    [
                        [0.5333, 0.5333, 0.5333, 0.5333, 0.5333, 0.5333],
                        [0.5333, 0.5052, 0.1167, 0.1636, 0.2603, 0.5333],
                        [0.5333, 0.8098, 0.2663, 0.2481, 0.0000, 0.5333],
                        [0.5333, 1.0000, 0.7210, 0.5643, 0.0701, 0.5333],
                        [0.5333, 0.7648, 0.1161, 0.0098, 0.3040, 0.5333],
                        [0.5333, 0.5333, 0.5333, 0.5333, 0.5333, 0.5333],
                    ]
                ),
            ).is_close()
            assert equality.EqualityTorch(instance[0][2], coords_x1_truth).is_close()
            assert equality.EqualityTorch(instance[0][3], coords_x2_truth).is_close()

        for instance in ds.dataset_masked:
            assert equality.EqualityTorch(
                instance[0][0],
                torch.tensor(
                    [
                        [0.3683, 0.3683, 0.3683, 0.3683, 7.0000, 0.3683],
                        [0.3683, 0.8110, 0.7109, 0.8207, 0.4728, 0.3683],
                        [0.3683, 0.7537, 0.8019, 0.6743, 7.0000, 0.3683],
                        [0.3683, 0.9627, 1.0000, 0.7434, 7.0000, 0.3683],
                        [0.3683, 7.0000, 0.7301, 0.7690, 0.7184, 7.0000],
                        [7.0000, 0.3683, 0.3683, 0.3683, 0.3683, 7.0000],
                    ]
                ),
            ).is_close()
            assert equality.EqualityTorch(
                instance[0][1],
                torch.tensor(
                    [
                        [0.5333, 0.5333, 7.0000, 7.0000, 0.5333, 0.5333],
                        [0.5333, 0.5052, 0.1167, 0.1636, 0.2603, 0.5333],
                        [0.5333, 0.8098, 7.0000, 0.2481, 0.0000, 0.5333],
                        [0.5333, 1.0000, 7.0000, 0.5643, 0.0701, 7.0000],
                        [0.5333, 0.7648, 0.1161, 0.0098, 0.3040, 0.5333],
                        [7.0000, 0.5333, 0.5333, 7.0000, 0.5333, 0.5333],
                    ]
                ),
            ).is_close()
            assert equality.EqualityTorch(instance[0][2], coords_x1_truth).is_close()
            assert equality.EqualityTorch(instance[0][3], coords_x2_truth).is_close()

        ds.remask()
        for instance in ds.dataset_masked:
            assert equality.EqualityTorch(
                instance[0][0],
                torch.tensor(
                    [
                        [0.3683, 0.3683, 7.0000, 0.3683, 0.3683, 7.0000],
                        [0.3683, 0.8110, 0.7109, 0.8207, 0.4728, 0.3683],
                        [7.0000, 0.7537, 0.8019, 0.6743, 0.0000, 0.3683],
                        [7.0000, 0.9627, 1.0000, 0.7434, 0.3539, 7.0000],
                        [7.0000, 0.7628, 0.7301, 0.7690, 0.7184, 0.3683],
                        [0.3683, 0.3683, 0.3683, 0.3683, 7.0000, 0.3683],
                    ]
                ),
            ).is_close()
            assert equality.EqualityTorch(
                instance[0][1],
                torch.tensor(
                    [
                        [0.5333, 0.5333, 0.5333, 7.0000, 0.5333, 0.5333],
                        [0.5333, 7.0000, 0.1167, 0.1636, 0.2603, 0.5333],
                        [7.0000, 0.8098, 7.0000, 7.0000, 0.0000, 7.0000],
                        [0.5333, 1.0000, 0.7210, 0.5643, 0.0701, 0.5333],
                        [0.5333, 7.0000, 0.1161, 0.0098, 0.3040, 0.5333],
                        [0.5333, 0.5333, 0.5333, 0.5333, 0.5333, 0.5333],
                    ]
                ),
            ).is_close()
            assert equality.EqualityTorch(instance[0][2], coords_x1_truth).is_close()
            assert equality.EqualityTorch(instance[0][3], coords_x2_truth).is_close()
