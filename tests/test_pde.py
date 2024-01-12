import pytest
import torch

from src.pde.pde import Distance, Grid, GridTime, GridTwoD
from src.util.equality import EqualityBuiltin


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


class TestGrid:
    def test_grid(self):
        with pytest.raises(ValueError):
            Grid(n_pts=0, stepsize=0.1, start=3)
            Grid(n_pts=10, stepsize=0, start=3)
            Grid(n_pts=10, stepsize=-0.1, start=3)

        assert EqualityBuiltin(
            list(Grid(n_pts=1, stepsize=0.1, start=3).step()),
            [3.0],
        ).is_equal()

        assert EqualityBuiltin(
            list(Grid(n_pts=5, stepsize=0.1, start=3).step()),
            [3.0, 3.1, 3.2, 3.3, 3.4],
        ).is_equal()


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
    def test_formatter(self):
        step = 3

        assert (
            GridTime(n_steps=1, stepsize=0.1).timestep_formatted(step)
            == GridTime(n_steps=9, stepsize=0.1).timestep_formatted(step)
            == "3"
        )
        assert (
            GridTime(n_steps=10, stepsize=0.1).timestep_formatted(step)
            == GridTime(n_steps=99, stepsize=0.1).timestep_formatted(step)
            == "03"
        )
        assert (
            GridTime(n_steps=100, stepsize=0.1).timestep_formatted(step)
            == GridTime(n_steps=999, stepsize=0.1).timestep_formatted(step)
            == "003"
        )

    def test_step(self):
        gr = GridTime(n_steps=50, stepsize=0.1)

        assert list(gr.step()) == list(range(1, 51))
