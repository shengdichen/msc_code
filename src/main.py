import logging

from src import problem
from src.definition import DEFINITION

logger = logging.getLogger(__name__)


class Pipeline:
    def work(self) -> None:
        self.plot_eval()

    def plot_mask(self) -> None:
        problem.ProblemMask().plot()

    def plot_raw(self) -> None:
        DEFINITION.seed(42)
        problem.ProblemPoisson().plot_raw()

        DEFINITION.seed(2)
        problem.ProblemHeat().plot_raw()

        DEFINITION.seed(37)
        problem.ProblemWave().plot_raw()

    def plot_eval(self) -> None:
        DEFINITION.seed(42)
        poisson = problem.ProblemPoisson()
        poisson.plot_error_single()
        poisson.plot_error_double()

        DEFINITION.seed(42)
        heat = problem.ProblemHeat()
        heat.plot_error_single()
        heat.plot_error_double()

        DEFINITION.seed(42)
        wave = problem.ProblemWave()
        wave.plot_error_single()
        wave.plot_error_double()

    def dynamic_remask(self) -> None:
        DEFINITION.seed(42)
        poisson = problem.ProblemPoisson()
        poisson.plot_remask()

    def plot_reconstruction(self) -> None:
        DEFINITION.seed(42)
        poisson = problem.ProblemPoisson()
        poisson.plot_reconstruction()

    def island_ring(self) -> None:
        DEFINITION.seed(42)
        problem.ProblemPoisson().island_ring()

    def mask_nonstandard(self) -> None:
        DEFINITION.seed(42)
        problem.ProblemPoisson().mask_nonstandard()

        DEFINITION.seed(42)
        problem.ProblemWave().mask_nonstandard()


def main():
    DEFINITION.seed()
    DEFINITION.configure_font_matplotlib(font_latex=False)

    p = Pipeline()
    p.work()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(module)s: [%(levelname)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y.%m.%d-%H:%M:%S",
    )
    main()
