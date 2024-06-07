import logging

from src import problem
from src.definition import DEFINITION

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self):
        self._problem_poisson = problem.ProblemPoisson()
        self._problem_heat = problem.ProblemHeat()
        self._problem_wave = problem.ProblemWave()

        self._problems = [self._problem_poisson, self._problem_heat, self._problem_wave]

    def work(self) -> None:
        for pr in self._problems:
            pr.eval()


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
