import math

import numpy as np
import pytest
import torch

from src.numerics import distance, grid
from src.numerics.equality import EqualityBuiltin, EqualityTorch


class TestDistance:
    def test_mse(self):
        assert torch.equal(distance.Distance(5).mse(), torch.tensor(25))
        assert torch.equal(distance.Distance(5, 0).mse(), torch.tensor(25))

        assert torch.equal(distance.Distance(5, 1).mse(), torch.tensor(16))
        assert torch.equal(distance.Distance(5, -1).mse(), torch.tensor(36))

        assert torch.equal(
            distance.Distance(torch.tensor([1] * 3), 1).mse(), torch.tensor(0)
        )
        assert torch.equal(
            distance.Distance(torch.tensor([2, 3, 4, 5], dtype=torch.float), 1).mse(),
            torch.tensor(7.5),  # (1 + 4 + 9 + 16) / 4
        )
        assert torch.equal(
            distance.Distance(
                torch.tensor([1, 4, 7], dtype=torch.float),
                torch.tensor([1, 2, 3], dtype=torch.float),
            ).mse(),
            torch.tensor(20 / 3),  # (0 + 4 + 16) / 3
        )

    def test_msc_relative(self):
        assert torch.equal(
            distance.Distance(
                torch.tensor([1, 2, 3], dtype=torch.float),
                torch.tensor([1, 2, 3], dtype=torch.float),
            ).mse_relative(),
            torch.tensor(0),
        )
        assert torch.equal(
            distance.Distance(
                torch.tensor([0, 0, 0], dtype=torch.float),
                torch.tensor([1, 2, 3], dtype=torch.float),
            ).mse_relative(),
            torch.tensor(1.0),
        )

        assert torch.equal(
            distance.Distance(
                torch.tensor([0, 0, 1], dtype=torch.float),
                torch.tensor([0, 0, 2], dtype=torch.float),
            ).mse_relative(),
            torch.tensor(0.5),
        )

        assert EqualityTorch(
            distance.Distance(
                torch.tensor([1, 4, 7], dtype=torch.float),
                torch.tensor([1, 2, 3], dtype=torch.float),
            ).mse_relative(),
            torch.tensor(1.1952),
        ).is_close()

    def test_mse_percentage(self):
        assert (
            distance.Distance(
                torch.tensor([1, 2, 3], dtype=torch.float),
                torch.tensor([1, 2, 3], dtype=torch.float),
            ).mse_percentage()
            == "0.0000%"
        )
        assert (
            distance.Distance(
                torch.tensor([0, 0, 0], dtype=torch.float),
                torch.tensor([1, 2, 3], dtype=torch.float),
            ).mse_percentage()
            == "100.0000%"
        )

        assert (
            distance.Distance(
                torch.tensor([0, 0, 1], dtype=torch.float),
                torch.tensor([0, 0, 2], dtype=torch.float),
            ).mse_percentage()
            == "50.0000%"
        )

        assert (
            distance.Distance(
                torch.tensor([1, 4, 7], dtype=torch.float),
                torch.tensor([1, 2, 3], dtype=torch.float),
            ).mse_percentage()
            == "119.5229%"
        )


class TestGrid:
    def test_init_error(self):
        with pytest.raises(ValueError):
            grid.Grid(n_pts=0, stepsize=0.1, start=3)
            grid.Grid(n_pts=10, stepsize=0, start=3)
            grid.Grid(n_pts=10, stepsize=-0.1, start=3)

    def test_step(self):
        gr = grid.Grid(n_pts=1, stepsize=0.1, start=3)
        assert EqualityBuiltin(
            gr.step(),
            [3.0],
        ).is_equal()
        assert not gr.step(with_start=False)

        gr = grid.Grid(n_pts=5, stepsize=0.1, start=3)
        assert EqualityBuiltin(
            list(gr.step()),
            [3.0, 3.1, 3.2, 3.3, 3.4],
        ).is_equal()
        assert EqualityBuiltin(
            list(gr.step(with_start=False)),
            [3.1, 3.2, 3.3, 3.4],
        ).is_equal()
        assert EqualityBuiltin(
            list(gr.step(with_end=False)),
            [3.0, 3.1, 3.2, 3.3],
        ).is_equal()
        assert EqualityBuiltin(
            list(gr.step(with_start=False, with_end=False)),
            [3.1, 3.2, 3.3],
        ).is_equal()

    def test_step_with_index(self):
        gr = grid.Grid(n_pts=5, stepsize=0.1, start=3)

        for idx_val_actual, idx_val_expected in zip(
            gr.step_with_index(),
            [
                (0, 3.0),
                (1, 3.1),
                (2, 3.2),
                (3, 3.3),
                (4, 3.4),
            ],
        ):
            assert idx_val_actual[0] == idx_val_expected[0]
            assert math.isclose(idx_val_actual[1], idx_val_expected[1])
        for idx_val_actual, idx_val_expected in zip(
            gr.step_with_index(with_start=False),
            [
                (1, 3.1),
                (2, 3.2),
                (3, 3.3),
                (4, 3.4),
            ],
        ):
            assert idx_val_actual[0] == idx_val_expected[0]
            assert math.isclose(idx_val_actual[1], idx_val_expected[1])
        for idx_val_actual, idx_val_expected in zip(
            gr.step_with_index(with_end=False),
            [
                (0, 3.0),
                (1, 3.1),
                (2, 3.2),
                (3, 3.3),
            ],
        ):
            assert idx_val_actual[0] == idx_val_expected[0]
            assert math.isclose(idx_val_actual[1], idx_val_expected[1])
        for idx_val_actual, idx_val_expected in zip(
            gr.step_with_index(with_start=False, with_end=False),
            [
                (1, 3.1),
                (2, 3.2),
                (3, 3.3),
            ],
        ):
            assert idx_val_actual[0] == idx_val_expected[0]
            assert math.isclose(idx_val_actual[1], idx_val_expected[1])

    def test_boundary(self):
        gr = grid.Grid(n_pts=5, stepsize=0.1, start=3)
        assert gr.is_on_boundary(3.0)
        assert gr.is_on_boundary(3.4)

        assert not gr.is_on_boundary(3.1)
        assert not gr.is_on_boundary(3.2)
        assert not gr.is_on_boundary(3.3)

    def test_min_max_perc(self):
        gr = grid.Grid(n_pts=50, stepsize=0.1, start=3)

        assert (
            gr.min_max_perc() == gr.min_max_perc(from_start=0, to_end=0) == (3.0, 7.9)
        )

        assert gr.min_max_perc(from_start=0.1) == (3.49, 7.9)
        assert gr.min_max_perc(to_end=0.1) == (3.0, 7.41)
        assert gr.min_max_perc(from_start=0.5) == (5.45, 7.9)
        assert gr.min_max_perc(to_end=0.5) == (3.0, 5.45)

        assert EqualityBuiltin(
            gr.min_max_perc(from_start=0.1, to_end=0.1), (3.49, 7.41)
        ).is_close()
        assert gr.min_max_perc(from_start=0.5, to_end=0.5) == (5.45, 5.45)

        with pytest.raises(ValueError):
            gr.min_max_perc(from_start=0.6, to_end=0.5)
            gr.min_max_perc(from_start=0.5, to_end=0.6)

    def test_min_max_n_pts(self):
        gr = grid.Grid(n_pts=50, stepsize=0.1, start=3)

        assert (
            gr.min_max_n_pts() == gr.min_max_n_pts(from_start=0, to_end=0) == (3.0, 7.9)
        )

        assert (
            gr.min_max_n_pts(from_start=5)
            == gr.min_max_n_pts(from_start=0.1)
            == (3.5, 7.9)
        )
        assert gr.min_max_n_pts(to_end=5) == gr.min_max_n_pts(to_end=0.1) == (3.0, 7.4)
        assert (
            gr.min_max_n_pts(from_start=25)
            == gr.min_max_n_pts(from_start=0.5)
            == (5.5, 7.9)
        )
        assert gr.min_max_n_pts(to_end=25) == gr.min_max_n_pts(to_end=0.5) == (3.0, 5.4)

        assert (
            gr.min_max_n_pts(from_start=5, to_end=5)
            == gr.min_max_n_pts(from_start=0.1, to_end=0.1)
            == (3.5, 7.4)
        )

        with pytest.raises(ValueError):
            gr.min_max_n_pts(from_start=25, to_end=25)
            gr.min_max_n_pts(from_start=30, to_end=25)
            gr.min_max_n_pts(from_start=25, to_end=30)

    def test_sample_uniform(self) -> None:
        gr = grid.Grid(n_pts=50, stepsize=0.1, start=3)
        assert np.allclose(
            gr.sample_uniform(size=2, rng=np.random.default_rng(42)),
            np.array([6.79238464, 5.15050435]),
        )


class TestGrids:
    def test_indexes(self):
        gr_1 = grid.Grid(n_pts=3, stepsize=0.1, start=3)
        gr_2 = grid.Grid(n_pts=4, stepsize=0.1, start=4)
        grs = grid.Grids([gr_1, gr_2])
        assert list(grs.indexes()) == [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
        ]

    def test_sample(self):
        gr_1 = grid.Grid(n_pts=4, stepsize=0.1, start=3)
        gr_2 = grid.Grid(n_pts=5, stepsize=0.1, start=4)
        grs = grid.Grids([gr_1, gr_2])

        for _ in range(3):  # also test auto engine-reset after every draw
            assert EqualityTorch(
                (grs.samples_sobol(10)),
                torch.tensor(
                    [
                        [3.0000, 4.0000],
                        [3.1500, 4.2000],
                        [3.2250, 4.1000],
                        [3.0750, 4.3000],
                        [3.1125, 4.1500],
                        [3.2625, 4.3500],
                        [3.1875, 4.0500],
                        [3.0375, 4.2500],
                        [3.0563, 4.1250],
                        [3.2062, 4.3250],
                    ]
                ),
            ).is_close()

    def test_steps_no_index(self):
        gr_1 = grid.Grid(n_pts=4, stepsize=0.1, start=3)
        gr_2 = grid.Grid(n_pts=4, stepsize=0.1, start=4)
        gr_3 = grid.Grid(n_pts=4, stepsize=0.1, start=5)
        grs = grid.Grids([gr_1, gr_2, gr_3])

        assert len(list(grs.steps())) == 64

        boundaries = list(grs.boundaries())
        assert len(boundaries) == 56
        for vals in [
            (3, 4.1, 5.1),
            (3.3, 4.1, 5.1),
            (3.1, 4, 5.1),
            (3.1, 4.3, 5.1),
            (3.1, 4.1, 5),
            (3.1, 4.1, 5.3),
        ]:
            assert vals in boundaries

        internals = list(grs.internals())
        assert len(internals) == 8
        for vals in [
            (3.1, 4.1, 5.1),
            (3.2, 4.1, 5.1),
            (3.1, 4.2, 5.1),
            (3.1, 4.1, 5.2),
        ]:
            assert vals in internals

    def test_steps_with_index(self):
        gr_1 = grid.Grid(n_pts=4, stepsize=0.1, start=3)
        gr_2 = grid.Grid(n_pts=4, stepsize=0.1, start=4)
        grs = grid.Grids([gr_1, gr_2])

        assert list(grs.steps_with_index()) == [
            ((0, 3.0), (0, 4.0)),
            ((0, 3.0), (1, 4.1)),
            ((0, 3.0), (2, 4.2)),
            ((0, 3.0), (3, 4.3)),
            ((1, 3.1), (0, 4.0)),
            ((1, 3.1), (1, 4.1)),
            ((1, 3.1), (2, 4.2)),
            ((1, 3.1), (3, 4.3)),
            ((2, 3.2), (0, 4.0)),
            ((2, 3.2), (1, 4.1)),
            ((2, 3.2), (2, 4.2)),
            ((2, 3.2), (3, 4.3)),
            ((3, 3.3), (0, 4.0)),
            ((3, 3.3), (1, 4.1)),
            ((3, 3.3), (2, 4.2)),
            ((3, 3.3), (3, 4.3)),
        ]

        assert list(grs.boundaries_with_index()) == [
            ((0, 3.0), (0, 4.0)),
            ((0, 3.0), (1, 4.1)),
            ((0, 3.0), (2, 4.2)),
            ((0, 3.0), (3, 4.3)),
            ((1, 3.1), (0, 4.0)),
            ((1, 3.1), (3, 4.3)),
            ((2, 3.2), (0, 4.0)),
            ((2, 3.2), (3, 4.3)),
            ((3, 3.3), (0, 4.0)),
            ((3, 3.3), (1, 4.1)),
            ((3, 3.3), (2, 4.2)),
            ((3, 3.3), (3, 4.3)),
        ]
        assert list(grs.internals_with_index()) == [
            ((1, 3.1), (1, 4.1)),
            ((1, 3.1), (2, 4.2)),
            ((2, 3.2), (1, 4.1)),
            ((2, 3.2), (2, 4.2)),
        ]

    def test_is_on_boundary(self):
        gr_1 = grid.Grid(n_pts=4, stepsize=0.1, start=3)
        gr_2 = grid.Grid(n_pts=4, stepsize=0.1, start=4)
        gr_3 = grid.Grid(n_pts=4, stepsize=0.1, start=5)
        grs = grid.Grids([gr_1, gr_2, gr_3])

        assert grs.is_on_boundary([3, 4.1, 5.1])
        assert grs.is_on_boundary([3.3, 4.1, 5.1])
        assert grs.is_on_boundary([3.1, 4, 5.1])
        assert grs.is_on_boundary([3.1, 4.3, 5.1])
        assert grs.is_on_boundary([3.1, 4.1, 5])
        assert grs.is_on_boundary([3.1, 4.1, 5.3])

        assert not grs.is_on_boundary([3.1, 4.1, 5.1])
        assert not grs.is_on_boundary([3.2, 4.1, 5.1])
        assert not grs.is_on_boundary([3.1, 4.2, 5.1])
        assert not grs.is_on_boundary([3.1, 4.1, 5.2])

    def test_constants(self):
        gr_1 = grid.Grid(n_pts=4, stepsize=0.1)
        gr_2 = grid.Grid(n_pts=5, stepsize=0.1)
        grs = grid.Grids([gr_1, gr_2])

        assert torch.equal(grs.zeroes_like(), torch.zeros((4, 5)))
        assert torch.equal(grs.constants_like(0.0), torch.zeros((4, 5)))

        assert torch.equal(grs.constants_like(), torch.ones((4, 5)))
        assert torch.equal(grs.constants_like(42), 42 * torch.ones((4, 5)))

        assert np.allclose(grs.zeroes_like_numpy(), np.zeros((4, 5)))

    def test_as_mesh(self):
        gr_1 = grid.Grid(n_pts=4, stepsize=0.1, start=3)
        gr_2 = grid.Grid(n_pts=4, stepsize=0.1, start=4)
        grs = grid.Grids([gr_1, gr_2])

        coords = grs.coords_as_mesh()
        assert np.allclose(
            coords[0],
            [
                [3.0, 3.0, 3.0, 3.0],
                [3.1, 3.1, 3.1, 3.1],
                [3.2, 3.2, 3.2, 3.2],
                [3.3, 3.3, 3.3, 3.3],
            ],
        )
        assert np.allclose(
            coords[1],
            [
                [4.0, 4.1, 4.2, 4.3],
                [4.0, 4.1, 4.2, 4.3],
                [4.0, 4.1, 4.2, 4.3],
                [4.0, 4.1, 4.2, 4.3],
            ],
        )

        coords = grs.coords_as_mesh(indexing_machine_like=False)
        assert np.allclose(
            coords[0],
            [
                [3.0, 3.1, 3.2, 3.3],
                [3.0, 3.1, 3.2, 3.3],
                [3.0, 3.1, 3.2, 3.3],
                [3.0, 3.1, 3.2, 3.3],
            ],
        )
        assert np.allclose(
            coords[1],
            [
                [4.0, 4.0, 4.0, 4.0],
                [4.1, 4.1, 4.1, 4.1],
                [4.2, 4.2, 4.2, 4.2],
                [4.3, 4.3, 4.3, 4.3],
            ],
        )

    def test_flatten(self):
        gr_1 = grid.Grid(n_pts=5, stepsize=0.1, start=3)
        gr_2 = grid.Grid(n_pts=4, stepsize=0.1, start=4)
        grs = grid.Grids([gr_1, gr_2])

        target_unflattened = torch.tensor(
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [10, 20, 30, 40],
                [5, 6, 7, 8],
                [50, 60, 70, 80],
            ]
        )
        target_flattened = torch.tensor(
            [1, 2, 3, 4, 1, 2, 3, 4, 10, 20, 30, 40, 5, 6, 7, 8, 50, 60, 70, 80]
        )

        assert torch.allclose(
            grs.flattten(target_unflattened),
            target_flattened,
        )
        assert torch.allclose(grs.unflatten_2d(target_flattened), target_unflattened)

    def test_mask(self):
        gr_1 = grid.Grid(n_pts=5, stepsize=0.1, start=3)
        gr_2 = grid.Grid(n_pts=6, stepsize=0.1, start=4)
        grs = grid.Grids([gr_1, gr_2])

        target_raw = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
                [10, 11, 12, 13, 14, 15],
                [20, 21, 22, 23, 24, 25],
                [30, 31, 32, 33, 34, 35],
                [40, 41, 42, 43, 44, 45],
            ]
        )

        assert torch.allclose(
            grs.mask(target_raw, idx_min=1, idx_max=3),
            torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 11, 12, 13, 0, 0],
                    [0, 21, 22, 23, 0, 0],
                    [0, 31, 32, 33, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
        )

    def test_sample_uniform(self) -> None:
        gr_1 = grid.Grid(n_pts=10, stepsize=0.1, start=-100)
        gr_2 = grid.Grid(n_pts=10, stepsize=0.1, start=0)
        gr_3 = grid.Grid(n_pts=10, stepsize=0.1, start=+100)
        grs = grid.Grids([gr_1, gr_2, gr_3])

        assert np.allclose(
            grs.sample_uniform(rng=np.random.default_rng(42)),
            [[-99.30343956, 0.3949906, 100.77273813]],
        )
        assert np.allclose(
            (grs.sample_uniform(size=2, rng=np.random.default_rng(42))),
            [
                [-99.30343956, 0.77273813, 100.08475961],
                [-99.6050094, 0.62763123, 100.87806012],
            ],
        )


class TestGridTime:
    def test_n_pts(self):
        assert grid.GridTime(n_pts=100, stepsize=0.1).n_pts == 100

    def test_formatter(self):
        step = 3

        assert (
            grid.GridTime(n_pts=1, stepsize=0.1).timestep_formatted(step)
            == grid.GridTime(n_pts=9, stepsize=0.1).timestep_formatted(step)
            == "3"
        )
        assert (
            grid.GridTime(n_pts=10, stepsize=0.1).timestep_formatted(step)
            == grid.GridTime(n_pts=99, stepsize=0.1).timestep_formatted(step)
            == "03"
        )
        assert (
            grid.GridTime(n_pts=100, stepsize=0.1).timestep_formatted(step)
            == grid.GridTime(n_pts=999, stepsize=0.1).timestep_formatted(step)
            == "003"
        )

    def test_step(self):
        gr = grid.GridTime(n_pts=5, stepsize=0.1)

        assert EqualityBuiltin(
            list(gr.step()),
            [0.1, 0.2, 0.3, 0.4, 0.5],
        ).is_equal()
        assert EqualityBuiltin(
            list(gr.step(with_start=True)),
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        ).is_equal()
        assert EqualityBuiltin(
            list(gr.step(with_end=False)),
            [0.1, 0.2, 0.3, 0.4],
        ).is_equal()
        assert EqualityBuiltin(
            list(gr.step(with_start=True, with_end=False)),
            [0.0, 0.1, 0.2, 0.3, 0.4],
        ).is_equal()

    def test_step_with_index(self):
        gr = grid.GridTime(n_pts=5, stepsize=0.1)

        for idx_val_actual, idx_val_expected in zip(
            gr.step_with_index(),
            [
                (1, 0.1),
                (2, 0.2),
                (3, 0.3),
                (4, 0.4),
            ],
        ):
            assert idx_val_actual[0] == idx_val_expected[0]
            assert math.isclose(idx_val_actual[1], idx_val_expected[1])
        for idx_val_actual, idx_val_expected in zip(
            gr.step_with_index(with_start=True),
            [
                (0, 0.0),
                (1, 0.1),
                (2, 0.2),
                (3, 0.3),
                (4, 0.4),
            ],
        ):
            assert idx_val_actual[0] == idx_val_expected[0]
            assert math.isclose(idx_val_actual[1], idx_val_expected[1])
        for idx_val_actual, idx_val_expected in zip(
            gr.step_with_index(with_end=False),
            [
                (1, 0.1),
                (2, 0.2),
                (3, 0.3),
            ],
        ):
            assert idx_val_actual[0] == idx_val_expected[0]
            assert math.isclose(idx_val_actual[1], idx_val_expected[1])
        for idx_val_actual, idx_val_expected in zip(
            gr.step_with_index(with_start=True, with_end=False),
            [
                (0, 0.0),
                (1, 0.1),
                (2, 0.2),
                (3, 0.3),
            ],
        ):
            assert idx_val_actual[0] == idx_val_expected[0]
            assert math.isclose(idx_val_actual[1], idx_val_expected[1])

    def test_is_in(self):
        gr = grid.GridTime(n_pts=5, stepsize=0.1)

        for val in [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
        ]:
            assert gr.is_in(val)

    def test_index_of(self):
        gr = grid.GridTime(n_pts=5, stepsize=0.1)

        for idx_theirs, val in [
            (0, 0.0),
            (1, 0.1),
            (2, 0.2),
            (3, 0.3),
            (4, 0.4),
        ]:
            assert gr.index_of(val) == idx_theirs

    def test_is_init(self):
        gr = grid.GridTime(n_pts=5, stepsize=0.1)

        assert gr.is_init(0)
        assert not gr.is_init(0.1)
