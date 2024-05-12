import abc
import collections
import logging
import math
import pathlib
import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.deepl import factory
from src.numerics import grid
from src.pde import dataset, model
from src.pde.heat import dataset as dataset_heat
from src.pde.poisson import dataset as dataset_poisson
from src.pde.wave import dataset as dataset_wave
from src.util import dataset as util_dataset

logger = logging.getLogger(__name__)


class Problem:
    def __init__(self, n_channels_raw: int):
        # 2 (extra) channels each for (raw-)coords, sin-coords, cos-coords
        self._n_channels_lhs = n_channels_raw + 6

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

        self._datasets_train_single: list[  # type: ignore [annotation-unchecked]
            dataset.DatasetMaskedSingle
        ] = []
        self._datasets_train_double: list[  # type: ignore [annotation-unchecked]
            dataset.DatasetMaskedDouble
        ] = []

        self._intensities_eval = [(i * 10 + 5) for i in range(10)]  # in percentage
        self._datasets_evals_single_random: list[  # type: ignore [annotation-unchecked]
            dataset.DatasetMaskedSingle
        ] = []
        self._datasets_evals_double_random: list[  # type: ignore [annotation-unchecked]
            dataset.DatasetMaskedDouble
        ] = []
        self._datasets_evals_single_island: list[  # type: ignore [annotation-unchecked]
            dataset.DatasetMaskedSingle
        ] = []
        self._datasets_evals_double_island: list[  # type: ignore [annotation-unchecked]
            dataset.DatasetMaskedDouble
        ] = []
        self._load_datasets()

    def _load_datasets(self) -> None:
        mask: util_dataset.Masker

        for mask in self._masks_train:
            ds_single = self._dataset_single(mask)
            ds_single.as_train(self._n_instances_train)
            self._datasets_train_single.append(ds_single)
            ds_double = self._dataset_double(mask)
            ds_double.as_train(self._n_instances_train)
            self._datasets_train_double.append(ds_double)

        for mask in [
            util_dataset.MaskerRandom.from_min_max(
                intensity_min=i / 10, intensity_max=i / 10 + 0.1
            )
            for i in range(10)
        ]:
            ds_single = self._dataset_single(mask)
            ds_single.as_eval(self._n_instances_eval)
            self._datasets_evals_single_random.append(ds_single)
            ds_double = self._dataset_double(mask)
            ds_double.as_eval(self._n_instances_eval)
            self._datasets_evals_double_random.append(ds_double)

        for mask in [
            util_dataset.MaskerIsland.from_min_max(
                intensity_min=i / 10, intensity_max=i / 10 + 0.1
            )
            for i in range(10)
        ]:
            ds_single = self._dataset_single(mask)
            ds_single.as_eval(self._n_instances_eval)
            self._datasets_evals_single_island.append(ds_single)
            ds_double = self._dataset_double(mask)
            ds_double.as_eval(self._n_instances_eval)
            self._datasets_evals_double_island.append(ds_double)

    def train(self) -> None:
        self.train_single()
        self.train_double()

    def train_single(self) -> None:
        for ds_train in self._datasets_train_single:
            for m in self._models_current_single(ds_train):
                ds_eval = self._dataset_single(ds_train.masks[0])
                ds_eval.as_eval(self._n_instances_eval)
                m.datasets_eval = [ds_eval]
                m.train()

    def train_double(self) -> None:
        for ds_train in self._datasets_train_double:
            for m in self._models_current_double(ds_train):
                ds_eval = self._dataset_double(ds_train.masks[0])
                ds_eval.as_eval(self._n_instances_eval)
                m.datasets_eval = [ds_eval]
                m.train()

    def eval(self) -> None:
        self.eval_single()
        self.eval_double()

    def eval_single(self) -> None:
        for m in self.models_single():
            m.load_network()
            m.datasets_eval = self._datasets_evals_single_random
            m.eval(print_result=True)
            m.datasets_eval = self._datasets_evals_single_island
            m.eval(print_result=True)

    def eval_double(self) -> None:
        for m in self.models_double():
            m.load_network()
            m.datasets_eval = self._datasets_evals_double_random
            m.eval(print_result=True)
            m.datasets_eval = self._datasets_evals_double_island
            m.eval(print_result=True)

    def plot_error_single(self) -> None:
        y_max, y_clip = 0.18, 0.175
        for ds_train in self._datasets_train_single:
            models = list(self._models_current_single(ds_train))
            fig, (ax_random, ax_island) = plt.subplots(1, 2, figsize=(11, 5.5), dpi=200)
            for m in models:
                m.load_network()
                m.datasets_eval = self._datasets_evals_single_random
                errors_random = m.errors(clip_at_max=y_clip)
                m.datasets_eval = self._datasets_evals_single_island
                errors_island = m.errors(clip_at_max=y_clip)
                self._plot_errors(
                    ax_random,
                    errors_random,
                    ax_island,
                    errors_island,
                    label=m.name_network,
                )

            self._style_plot(
                ax_random,
                ax_island,
                mask_min_max=ds_train.masks[0].min_max(as_percentage=True),
                y_max=y_max * 100,
                y_ticks=[2.5, 5, 10, 15, 17.5],
            )

            fig.suptitle(ds_train.name_human(multiline=True))
            ax_random.set_title("(Eval-)Masking: Random")
            ax_island.set_title("(Eval-)Masking: Island")
            ax_random.set_ylabel("error [$L^2$]")

            path = pathlib.Path(f"{ds_train.path}.png")
            fig.savefig(path)
            plt.close(fig)

    def plot_error_double(self) -> None:
        y_max, y_clip = 0.26, 0.25
        for ds_train in self._datasets_train_double:
            models = list(self._models_current_double(ds_train))
            fig, ((ax_0_random, ax_0_island), (ax_1_random, ax_1_island)) = (
                plt.subplots(2, 2, figsize=(11, 11), dpi=200)
            )
            for m in models:
                m.load_network()
                m.datasets_eval = self._datasets_evals_double_random
                errors_0_random, errors_1_random = m.errors(clip_at_max=y_clip)
                m.datasets_eval = self._datasets_evals_double_island
                errors_0_island, errors_1_island = m.errors(clip_at_max=y_clip)
                self._plot_errors(
                    ax_0_random,
                    errors_0_random,
                    ax_0_island,
                    errors_0_island,
                    label=m.name_network,
                )
                self._plot_errors(
                    ax_1_random,
                    errors_1_random,
                    ax_1_island,
                    errors_1_island,
                    label=m.name_network,
                )

            for ax_random, ax_island in (
                (ax_0_random, ax_0_island),
                (ax_1_random, ax_1_island),
            ):
                self._style_plot(
                    ax_random,
                    ax_island,
                    mask_min_max=ds_train.masks[0].min_max(as_percentage=True),
                    y_max=y_max * 100,
                    y_ticks=[i * 5 for i in range(6)],
                    baseline_high=20.0,
                )

            fig.suptitle(ds_train.name_human(multiline=True))
            ax_0_random.set_title("Channel 0 Masking: Random")
            ax_0_island.set_title("Channel 0 Masking: Island")
            ax_1_random.set_title("Channel 1 Masking: Random")
            ax_1_island.set_title("Channel 1 Masking: Island")
            ax_0_random.set_ylabel("error [$L^2$]")
            ax_1_random.set_ylabel("error [$L^2$]")

            path = pathlib.Path(f"{ds_train.path}.png")
            fig.savefig(path)
            plt.close(fig)

    def _plot_errors(
        self,
        ax_random: mpl.axes.Axes,
        errors_random: np.ndarray,
        ax_island: mpl.axes.Axes,
        errors_island: np.ndarray,
        label: str,
    ) -> None:
        style = {"linestyle": "dashed", "linewidth": 1.0, "marker": "x"}
        ax_random.plot(
            self._intensities_eval,
            errors_random,
            **style,
            label=label,
        )
        ax_island.plot(
            self._intensities_eval,
            errors_island,
            **style,
            label=label,
        )

    def _style_plot(
        self,
        ax_random: mpl.axes.Axes,
        ax_island: mpl.axes.Axes,
        mask_min_max: tuple[float, float],
        y_max: float,
        y_ticks: typing.Sequence[float],
        baseline_low: float = 5.0,
        baseline_high: float = 15.0,
    ) -> None:
        for ax in [ax_random, ax_island]:
            ax.legend()

            ax.set_xlabel("masking intensity")
            ax.set_xlim(-3, 103)
            ax.set_xticks([10 * (x + 1) for x in range(9)])
            ax.tick_params(axis="x", labelrotation=30, labelsize="small")
            if not mask_min_max == (0, 100):
                ax.fill_betweenx(
                    range(math.ceil(y_max) + 1),
                    *mask_min_max,
                    color="lightgrey",
                )
            ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))

            ax.set_ylim(0.0, y_max)
            ax.set_yticks(y_ticks)
            ax.tick_params(axis="y", labelrotation=30, labelsize="small")
            ax.axhline(
                baseline_low,
                linestyle="dashed",
                color="darkgrey",
                linewidth=1.5,
            )
            ax.axhline(
                baseline_high,
                linestyle="dashed",
                color="darkgrey",
                linewidth=1.5,
            )
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=1))

    def models_single(self) -> collections.abc.Generator[model.ModelSingle, None, None]:
        for ds_train in self._datasets_train_single:
            yield from self._models_current_single(ds_train)

    def models_double(self) -> collections.abc.Generator[model.ModelDouble, None, None]:
        for ds_train in self._datasets_train_double:
            yield from self._models_current_double(ds_train)

    def _models_current_single(
        self, ds_train: dataset.DatasetMaskedSingle
    ) -> collections.abc.Generator[model.ModelSingle, None, None]:
        for network in factory.Network.all(dim_lhs=self._n_channels_lhs, dim_rhs=1):
            yield model.ModelSingle(network, ds_train)

    def _models_current_double(
        self, ds_train: dataset.DatasetMaskedDouble
    ) -> collections.abc.Generator[model.ModelDouble, None, None]:
        for network in factory.Network.all(dim_lhs=self._n_channels_lhs, dim_rhs=2):
            yield model.ModelDouble(network, ds_train)

    @abc.abstractmethod
    def _dataset_raw(self) -> dataset.DatasetPDE2d:
        raise NotImplementedError

    @abc.abstractmethod
    def _dataset_single(self, mask: util_dataset.Masker) -> dataset.DatasetMaskedSingle:
        raise NotImplementedError

    @abc.abstractmethod
    def _dataset_double(self, mask: util_dataset.Masker) -> dataset.DatasetMaskedDouble:
        raise NotImplementedError


class ProblemPoisson(Problem):
    def __init__(self):
        super().__init__(n_channels_raw=dataset_poisson.DatasetPoisson2d.N_CHANNELS)

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

    def _dataset_double(self, mask: util_dataset.Masker) -> dataset.DatasetMaskedDouble:
        return dataset_poisson.DatasetMaskedDoublePoisson(self._dataset_raw(), mask)


class ProblemHeat(Problem):
    def __init__(self):
        super().__init__(n_channels_raw=dataset_heat.DatasetHeat.N_CHANNELS)

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

    def _dataset_double(self, mask: util_dataset.Masker) -> dataset.DatasetMaskedDouble:
        return dataset_heat.DatasetMaskedDoubleHeat(self._dataset_raw(), mask)


class ProblemWave(Problem):
    def __init__(self):
        super().__init__(n_channels_raw=dataset_wave.DatasetWave.N_CHANNELS)

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

    def _dataset_double(self, mask: util_dataset.Masker) -> dataset.DatasetMaskedDouble:
        return dataset_wave.DatasetMaskedDoubleWave(self._dataset_raw(), mask)
