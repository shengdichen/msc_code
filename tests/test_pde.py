import math

import numpy as np
import pytest
import torch

from src.pde.pde import Distance, Grid, Grids, GridTime, GridTwoD
from src.util.equality import EqualityBuiltin, EqualityTorch


class TestDistance:
    def test_mse(self):
        assert torch.equal(Distance(5).mse(), torch.tensor(25))
        assert torch.equal(Distance(5, 0).mse(), torch.tensor(25))

        assert torch.equal(Distance(5, 1).mse(), torch.tensor(16))
        assert torch.equal(Distance(5, -1).mse(), torch.tensor(36))

        assert torch.equal(Distance(torch.tensor([1] * 3), 1).mse(), torch.tensor(0))
        assert torch.equal(
            Distance(torch.tensor([2, 3, 4, 5], dtype=torch.float), 1).mse(),
            torch.tensor(7.5),  # (1 + 4 + 9 + 16) / 4
        )
        assert torch.equal(
            Distance(
                torch.tensor([1, 4, 7], dtype=torch.float),
                torch.tensor([1, 2, 3], dtype=torch.float),
            ).mse(),
            torch.tensor(20 / 3),  # (0 + 4 + 16) / 3
        )

    def test_msc_relative(self):
        assert torch.equal(
            Distance(
                torch.tensor([1, 2, 3], dtype=torch.float),
                torch.tensor([1, 2, 3], dtype=torch.float),
            ).mse_relative(),
            torch.tensor(0),
        )
        assert torch.equal(
            Distance(
                torch.tensor([0, 0, 0], dtype=torch.float),
                torch.tensor([1, 2, 3], dtype=torch.float),
            ).mse_relative(),
            torch.tensor(1.0),
        )

        assert torch.equal(
            Distance(
                torch.tensor([0, 0, 1], dtype=torch.float),
                torch.tensor([0, 0, 2], dtype=torch.float),
            ).mse_relative(),
            torch.tensor(0.5),
        )

        assert EqualityTorch(
            Distance(
                torch.tensor([1, 4, 7], dtype=torch.float),
                torch.tensor([1, 2, 3], dtype=torch.float),
            ).mse_relative(),
            torch.tensor(1.1952),
        ).is_close()

    def test_mse_percentage(self):
        assert (
            Distance(
                torch.tensor([1, 2, 3], dtype=torch.float),
                torch.tensor([1, 2, 3], dtype=torch.float),
            ).mse_percentage()
            == "0.0000%"
        )
        assert (
            Distance(
                torch.tensor([0, 0, 0], dtype=torch.float),
                torch.tensor([1, 2, 3], dtype=torch.float),
            ).mse_percentage()
            == "100.0000%"
        )

        assert (
            Distance(
                torch.tensor([0, 0, 1], dtype=torch.float),
                torch.tensor([0, 0, 2], dtype=torch.float),
            ).mse_percentage()
            == "50.0000%"
        )

        assert (
            Distance(
                torch.tensor([1, 4, 7], dtype=torch.float),
                torch.tensor([1, 2, 3], dtype=torch.float),
            ).mse_percentage()
            == "119.5229%"
        )


class TestGrid:
    def test_init_error(self):
        with pytest.raises(ValueError):
            Grid(n_pts=0, stepsize=0.1, start=3)
            Grid(n_pts=10, stepsize=0, start=3)
            Grid(n_pts=10, stepsize=-0.1, start=3)

    def test_step(self):
        gr = Grid(n_pts=1, stepsize=0.1, start=3)
        assert EqualityBuiltin(
            gr.step(),
            [3.0],
        ).is_equal()
        assert not gr.step(with_start=False)

        gr = Grid(n_pts=5, stepsize=0.1, start=3)
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
        gr = Grid(n_pts=5, stepsize=0.1, start=3)

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
        gr = Grid(n_pts=5, stepsize=0.1, start=3)
        assert gr.is_on_boundary(3.0)
        assert gr.is_on_boundary(3.4)

        assert not gr.is_on_boundary(3.1)
        assert not gr.is_on_boundary(3.2)
        assert not gr.is_on_boundary(3.3)


class TestGrids:
    def test_steps_no_index(self):
        gr_1 = Grid(n_pts=4, stepsize=0.1, start=3)
        gr_2 = Grid(n_pts=4, stepsize=0.1, start=4)
        gr_3 = Grid(n_pts=4, stepsize=0.1, start=5)
        grs = Grids([gr_1, gr_2, gr_3])

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
        gr_1 = Grid(n_pts=4, stepsize=0.1, start=3)
        gr_2 = Grid(n_pts=4, stepsize=0.1, start=4)
        grs = Grids([gr_1, gr_2])

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
        gr_1 = Grid(n_pts=4, stepsize=0.1, start=3)
        gr_2 = Grid(n_pts=4, stepsize=0.1, start=4)
        gr_3 = Grid(n_pts=4, stepsize=0.1, start=5)
        grs = Grids([gr_1, gr_2, gr_3])

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

    def test_zeroes(self):
        gr_1 = Grid(n_pts=4, stepsize=0.1)
        gr_2 = Grid(n_pts=5, stepsize=0.1)
        grs = Grids([gr_1, gr_2])

        assert torch.equal(grs.zeroes_like(), torch.zeros((4, 5)))

    def test_as_mesh(self):
        gr_1 = Grid(n_pts=4, stepsize=0.1, start=3)
        gr_2 = Grid(n_pts=4, stepsize=0.1, start=4)
        grs = Grids([gr_1, gr_2])

        coords = grs.coords_as_mesh()
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


class TestGridTwoD:
    def test_dim(self):
        gr = GridTwoD(n_gridpts_x=50, n_gridpts_y=40, stepsize_x=0.1, step_size_y=0.15)
        assert (
            gr.coords_x.shape
            == gr.coords_y.shape
            == gr.init_solution_zeros().shape
            == (40, 50)
        )


class TestGridTime:
    def test_n_pts(self):
        assert GridTime(n_pts=100, stepsize=0.1).n_pts == 100

    def test_formatter(self):
        step = 3

        assert (
            GridTime(n_pts=1, stepsize=0.1).timestep_formatted(step)
            == GridTime(n_pts=9, stepsize=0.1).timestep_formatted(step)
            == "3"
        )
        assert (
            GridTime(n_pts=10, stepsize=0.1).timestep_formatted(step)
            == GridTime(n_pts=99, stepsize=0.1).timestep_formatted(step)
            == "03"
        )
        assert (
            GridTime(n_pts=100, stepsize=0.1).timestep_formatted(step)
            == GridTime(n_pts=999, stepsize=0.1).timestep_formatted(step)
            == "003"
        )

    def test_step(self):
        gr = GridTime(n_pts=5, stepsize=0.1)

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
        gr = GridTime(n_pts=5, stepsize=0.1)

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
        gr = GridTime(n_pts=5, stepsize=0.1)

        for val in [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
        ]:
            assert gr.is_in(val)

    def test_index_of(self):
        gr = GridTime(n_pts=5, stepsize=0.1)

        for idx_theirs, val in [
            (0, 0.0),
            (1, 0.1),
            (2, 0.2),
            (3, 0.3),
            (4, 0.4),
        ]:
            assert gr.index_of(val) == idx_theirs

    def test_is_init(self):
        gr = GridTime(n_pts=5, stepsize=0.1)

        assert gr.is_init(0)
        assert not gr.is_init(0.1)
