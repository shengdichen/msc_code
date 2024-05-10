import abc
import collections
import logging
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt

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

        self._datasets_train: list[  # type: ignore [annotation-unchecked]
            dataset.DatasetMaskedSingle
        ] = []

        self._intensities_eval = [(i * 10 + 5) for i in range(10)]  # in percentage
        self._datasets_evals_random: list[  # type: ignore [annotation-unchecked]
            dataset.DatasetMaskedSingle
        ] = []
        self._datasets_evals_island: list[  # type: ignore [annotation-unchecked]
            dataset.DatasetMaskedSingle
        ] = []
        self._load_datasets()

    def _load_datasets(self) -> None:
        for mask in self._masks_train:
            ds = self._dataset_single(mask)
            ds.as_train(self._n_instances_train)
            self._datasets_train.append(ds)

        for mask in [
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=i / 10, intensity_max=i / 10 + 0.1
            )
            for i in range(10)
        ]:
            ds = self._dataset_single(mask)
            ds.as_eval(self._n_instances_eval)
            self._datasets_evals_random.append(ds)

        for mask in [
            util_dataset.MaskerIsland.from_min_max(
                intensity_min=i / 10, intensity_max=i / 10 + 0.1
            )
            for i in range(10)
        ]:
            ds = self._dataset_single(mask)
            ds.as_eval(self._n_instances_eval)
            self._datasets_evals_island.append(ds)

    def train(self) -> None:
        for ds_train in self._datasets_train:
            for m in self._models_current(ds_train):
                ds_eval = self._dataset_single(ds_train.masks[0])
                ds_eval.as_eval(self._n_instances_eval)
                m.datasets_eval = [ds_eval]
                m.train()

    def eval(self) -> None:
        for m in self.models_single():
            m.load_network()
            m.datasets_eval = self._datasets_evals_random
            m.eval(print_result=True)
            m.datasets_eval = self._datasets_evals_island
            m.eval(print_result=True)

    def plot_error(self) -> None:
        for ds_train in self._datasets_train:
            models = list(self._models_current(ds_train))
            fig, (ax_random, ax_island) = plt.subplots(1, 2, figsize=(11, 5.5), dpi=200)
            style = {"linestyle": "dashed", "linewidth": 1.0, "marker": "x"}
            for m in models:
                m.load_network()
                m.datasets_eval = self._datasets_evals_random
                ax_random.plot(
                    self._intensities_eval,
                    m.errors(clip_at_max=0.175),
                    **style,
                    label=m.name_network,
                )
                m.datasets_eval = self._datasets_evals_island
                ax_island.plot(
                    self._intensities_eval,
                    m.errors(clip_at_max=0.175),
                    **style,
                    label=m.name_network,
                )

            for ax in [ax_random, ax_island]:
                ax.legend()

                y_max = 18

                ax.set_xlabel("masking intensity")
                ax.set_xlim(-3, 103)
                ax.set_xticks([10 * (x + 1) for x in range(9)])
                ax.tick_params(axis="x", labelrotation=30, labelsize="small")
                mask_min_max = ds_train.masks[0].min_max(as_percentage=True)
                if not mask_min_max == (0, 100):
                    ax.fill_betweenx(
                        range(y_max + 1),
                        *mask_min_max,
                        color="lightgrey",
                    )
                ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))

                ax.set_ylim(0.0, y_max)
                ax.set_yticks([2.5, 5, 10, 15, 17.5])
                ax.tick_params(axis="y", labelrotation=30, labelsize="small")
                ax.axhline(
                    5.0,
                    linestyle="dashed",
                    color="darkgrey",
                    linewidth=1.5,
                )
                ax.axhline(
                    15.0,
                    linestyle="dashed",
                    color="darkgrey",
                    linewidth=1.5,
                )
                ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=1))

            fig.suptitle(ds_train.name_human(multiline=True))
            ax_random.set_title("(Eval-)Masking: Random")
            ax_island.set_title("(Eval-)Masking: Island")
            ax_random.set_ylabel("error [$L^2$]")

            path = pathlib.Path(f"{ds_train.path}.png")
            fig.savefig(path)
            plt.close(fig)

    def models_single(self) -> collections.abc.Generator[model.Model, None, None]:
        for ds_train in self._datasets_train:
            yield from self._models_current(ds_train)

    def _models_current(
        self, ds_train: dataset.DatasetMaskedSingle
    ) -> collections.abc.Generator[model.Model, None, None]:
        for network in factory.Network.all(
            self._datasets_train[0].N_CHANNELS_LHS,
            self._datasets_train[0].N_CHANNELS_RHS,
        ):
            yield model.Model(network, ds_train)

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
