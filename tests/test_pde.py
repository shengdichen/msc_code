from src.pde.pde import GridTime, GridTwoD


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
