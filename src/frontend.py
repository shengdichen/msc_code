import logging

from src.deepl import factory
from src.numerics import grid
from src.pde import model
from src.pde.heat import dataset as dataset_heat
from src.pde.poisson import dataset as dataset_poisson
from src.pde.wave import dataset as dataset_wave
from src.util import dataset as util_dataset

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self):
        self._masks_train = [
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=0.0, intensity_max=1.0
            ),
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=0.0, intensity_max=0.5
            ),
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=0.5, intensity_max=1.0
            ),
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=0.25, intensity_max=0.75
            ),
        ]
        self._masks_eval_train = [
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=0.0, intensity_max=0.3
            ),
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=0.3, intensity_max=0.6
            ),
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=0.3, intensity_max=0.9
            ),
        ]

        self._masks_eval_full = [
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=i / 10, intensity_max=i / 10 + 0.1
            )
            for i in range(10)
        ]

        self._n_instances_eval, self._n_instances_train = 300, 1800

    def solve_poisson(self) -> None:
        grids = grid.Grids(
            [
                grid.Grid(n_pts=64, stepsize=0.01, start=0.0),
                grid.Grid(n_pts=64, stepsize=0.01, start=0.0),
            ]
        )
        raw = dataset_poisson.DatasetSin(grids)

        train = dataset_poisson.DatasetMaskedSinglePoisson(raw, self._masks_train[0])
        train.as_train(self._n_instances_train)
        evals = dataset_poisson.DatasetMaskedSinglePoisson.evals_from_masks(
            raw, self._masks_eval_train, self._n_instances_eval
        )

        for network in factory.Network.all(train.N_CHANNELS_LHS, train.N_CHANNELS_RHS):
            m = model.Model(network, train, evals)
            m.train()
            m.eval(print_result=True)

    def solve_heat(self) -> None:
        grids = grid.Grids(
            [
                grid.Grid.from_start_end(64, start=-1.0, end=1.0),
                grid.Grid.from_start_end(64, start=-1.0, end=1.0),
            ],
        )
        grid_time = grid.GridTime.from_start_end_only(end=0.005)
        raw = dataset_heat.DatasetHeat(grids, grid_time)

        train = dataset_heat.DatasetMaskedSingleHeat(raw, self._masks_train[0])
        train.as_train(self._n_instances_train)
        evals = dataset_heat.DatasetMaskedSingleHeat.evals_from_masks(
            raw, self._masks_eval_train, self._n_instances_eval
        )

        for network in factory.Network.all(train.N_CHANNELS_LHS, train.N_CHANNELS_RHS):
            m = model.Model(network, train, evals)
            m.train()
            m.eval(print_result=True)

    def solve_wave(self) -> None:
        grids = grid.Grids(
            [
                grid.Grid.from_start_end(64, start=0.0, end=1.0),
                grid.Grid.from_start_end(64, start=0.0, end=1.0),
            ],
        )
        grid_time = grid.GridTime.from_start_end_only(end=5.0)
        raw = dataset_wave.DatasetWave(grids, grid_time)

        train = dataset_wave.DatasetMaskedSingleWave(raw, self._masks_train[0])
        train.as_train(self._n_instances_train)
        evals = dataset_wave.DatasetMaskedSingleWave.evals_from_masks(
            raw, self._masks_eval_train, self._n_instances_eval
        )

        for network in factory.Network.all(train.N_CHANNELS_LHS, train.N_CHANNELS_RHS):
            m = model.Model(network, train, evals)
            m.train()
            m.eval(print_result=True)


def main():
    p = Pipeline()
    p.solve_poisson()
    p.solve_heat()
    p.solve_wave()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )
    main()
