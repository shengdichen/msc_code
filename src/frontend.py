import abc
import collections
import logging

from src.deepl import factory
from src.numerics import grid
from src.pde import dataset, model
from src.pde.heat import dataset as dataset_heat
from src.pde.poisson import dataset as dataset_poisson
from src.pde.wave import dataset as dataset_wave
from src.util import dataset as util_dataset

logger = logging.getLogger(__name__)


class Problem:
    def __init__(self):
        self._n_instances_eval, self._n_instances_train = 300, 1800

        self._masks_train = [
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=0.0, intensity_max=1.0
            ),
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=0.4, intensity_max=0.6
            ),
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=0.2, intensity_max=0.4
            ),
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=0.6, intensity_max=0.8
            ),
        ]

        self._masks_eval = [
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=i / 10, intensity_max=i / 10 + 0.1
            )
            for i in range(10)
        ]
        self._datasets_evals = []
        for mask in self._masks_eval:
            ds = self._dataset_single(mask)
            ds.as_eval(self._n_instances_eval)
            self._datasets_evals.append(ds)

    def train(self) -> None:
        for m in self.models_single():
            m.train()

    def eval(self) -> None:
        for m in self.models_single():
            m.load_network()
            m.datasets_eval = self._datasets_evals
            m.eval(print_result=True)

    def models_single(self) -> collections.abc.Generator[model.Model, None, None]:
        for mask in self._masks_train:
            ds_train = self._dataset_single(mask)
            ds_train.as_train(self._n_instances_train)
            ds_eval = self._dataset_single(mask)
            ds_eval.as_eval(self._n_instances_eval)

            for network in factory.Network.all(
                ds_train.N_CHANNELS_LHS, ds_train.N_CHANNELS_RHS
            ):
                yield model.Model(network, ds_train, [ds_eval])

    @abc.abstractmethod
    def _dataset_raw(self) -> dataset.DatasetPDE2d:
        raise NotImplementedError

    @abc.abstractmethod
    def _dataset_single(self, mask: util_dataset.Masker) -> dataset.DatasetMaskedSingle:
        raise NotImplementedError


class ProblemPoisson(Problem):
    def _dataset_raw(self) -> dataset.DatasetPDE2d:
        grids = grid.Grids(
            [
                grid.Grid(n_pts=64, stepsize=0.01, start=0.0),
                grid.Grid(n_pts=64, stepsize=0.01, start=0.0),
            ]
        )
        return dataset_poisson.DatasetSin(grids)

    def _dataset_single(self, mask: util_dataset.Masker) -> dataset.DatasetMaskedSingle:
        return dataset_poisson.DatasetMaskedSinglePoisson(self._dataset_raw(), mask)


class ProblemHeat(Problem):
    def _dataset_raw(self) -> dataset.DatasetPDE2d:
        grids = grid.Grids(
            [
                grid.Grid.from_start_end(64, start=-1.0, end=1.0),
                grid.Grid.from_start_end(64, start=-1.0, end=1.0),
            ],
        )
        grid_time = grid.GridTime.from_start_end_only(end=0.005)
        return dataset_heat.DatasetHeat(grids, grid_time)

    def _dataset_single(self, mask: util_dataset.Masker) -> dataset.DatasetMaskedSingle:
        return dataset_heat.DatasetMaskedSingleHeat(self._dataset_raw(), mask)


class ProblemWave(Problem):
    def _dataset_raw(self) -> dataset.DatasetPDE2d:
        grids = grid.Grids(
            [
                grid.Grid.from_start_end(64, start=0.0, end=1.0),
                grid.Grid.from_start_end(64, start=0.0, end=1.0),
            ],
        )
        grid_time = grid.GridTime.from_start_end_only(end=5.0)
        return dataset_wave.DatasetWave(grids, grid_time)

    def _dataset_single(self, mask: util_dataset.Masker) -> dataset.DatasetMaskedSingle:
        return dataset_wave.DatasetMaskedSingleWave(self._dataset_raw(), mask)


class Pipeline:
    def __init__(self):
        self._problem_poisson = ProblemPoisson()
        self._problem_heat = ProblemHeat()
        self._problem_wave = ProblemWave()

        self._problems = [self._problem_poisson, self._problem_heat, self._problem_wave]

    def work(self) -> None:
        for pr in self._problems:
            pr.eval()


def main():
    p = Pipeline()
    p.work()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )
    main()
