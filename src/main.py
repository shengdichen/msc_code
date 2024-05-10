import logging

from src import problem

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
    p = Pipeline()
    p.work()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s: [%(levelname)s] %(message)s", level=logging.DEBUG
    )
    main()
