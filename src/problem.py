import abc
import collections
import logging
import math
import pathlib
import pickle
import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.deepl import factory
from src.definition import DEFINITION
from src.numerics import grid
from src.pde import dataset, model
from src.pde.heat import dataset as dataset_heat
from src.pde.poisson import dataset as dataset_poisson
from src.pde.wave import dataset as dataset_wave
from src.util import dataset as util_dataset
from src.util import plot as util_plot

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
        self._datasets_evals_single_ring: list[  # type: ignore [annotation-unchecked]
            dataset.DatasetMaskedSingle
        ] = []
        self._datasets_evals_double_ring: list[  # type: ignore [annotation-unchecked]
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

        for mask in [
            util_dataset.MaskerRing.from_min_max(
                intensity_min=i / 10, intensity_max=i / 10 + 0.1
            )
            for i in range(10)
        ]:
            ds_single = self._dataset_single(mask)
            ds_single.as_eval(self._n_instances_eval)
            self._datasets_evals_single_ring.append(ds_single)
            ds_double = self._dataset_double(mask)
            ds_double.as_eval(self._n_instances_eval)
            self._datasets_evals_double_ring.append(ds_double)

    def train(self) -> None:
        self.train_single()
        self.train_double()

    def _plot_train_log(self, name_ds_train: str, name_model: str) -> None:
        DEFINITION.seed()
        ds_train = self._datasets_train_single[
            {"full": 0, "mid": 1, "low": 2, "high": 3}[name_ds_train]
        ]
        m = list(self._models_current_single(ds_train))[
            {"fno": 0, "cno": 1, "kno": 2, "unet": 3}[name_model]
        ]

        ds_eval = self._dataset_single(ds_train.masks[0])
        ds_eval.as_eval(self._n_instances_eval)
        m.datasets_eval = [ds_eval]

        n_remasks_stale_max = 3
        y_max, y_clip = 0.21, 0.21
        for n_epochs_stale_max in [30]:
            fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

            path = m.path_with_remask(n_epochs_stale_max)
            path_pickel = pathlib.Path(f"{path}.pickle")
            if not path_pickel.exists():
                res_train = m.train(
                    n_epochs_stale_max=n_epochs_stale_max,
                    n_remasks_stale_max=n_remasks_stale_max,
                    adhoc=True,
                )
                with open(path_pickel, "wb") as f:
                    pickle.dump(res_train, f)
            with open(path_pickel, "rb") as f:
                errors, epochs_remask, best_premask, bests = pickle.load(f)

            if not epochs_remask:
                logger.info("model> no plot-data, skipping")
                continue

            errors_non_perc = errors
            errors = 100 * np.clip(errors, a_min=0.0, a_max=y_clip)
            ax.plot(errors[:, 0], label="train")
            ax.plot(errors[:, 1], label="eval")
            ax.plot(
                best_premask[0],
                100 * best_premask[1],
                linestyle="",
                marker="^",
                markersize=9,
                color="red",
                label="best (pre-remask)",
            )
            ax.plot(
                bests[0],
                100 * np.array(bests[1]),
                linestyle="",
                marker="*",
                markersize=10,
                label="improvement (post-remask)",
            )

            best_pre = best_premask[1]
            print(f"best/pre> epoch: {best_premask[0]}, err-eval: {best_pre:.4%}")
            for epoch, err_eval in zip(bests[0], bests[1]):
                improv = err_eval / best_pre
                print(
                    f"best/post> epoch: {epoch}, err-eval: {err_eval:.4%}, improv: {improv:.3f}"
                )

            radius, ratio = 1.9, 5.5  # TODO: find a way to calc ratio
            width, height = ratio * radius, radius
            for coord_x in epochs_remask[:-1]:
                err_pre, err_post = (
                    errors_non_perc[coord_x - 1][0],
                    errors_non_perc[coord_x][0],
                )
                print(
                    f"remask> epoch: {coord_x}"
                    " | "
                    f"err-train-pre: {err_pre:.4%}; "
                    f"err-train-post: {err_post:.4%}; "
                    f"bump: {err_post/err_pre:.3f}"
                    " | "
                    f"err-eval-pre: {errors_non_perc[coord_x - 1][1]:.4%}; "
                    f"err-eval-post: {errors_non_perc[coord_x][1]:.4%}"
                )
                circle = mpl.patches.Ellipse(
                    (coord_x, 100 * err_post),
                    width,
                    height,
                    color="tab:blue",  # the default blue
                    linewidth=2,  # the default blue
                    fill=False,
                )
                ax.add_patch(circle)

            ax.set_xlabel("training epoch")
            ax.tick_params(axis="x", labelrotation=30, labelsize="small")
            for epoch in epochs_remask:
                ax.axvline(
                    epoch,
                    linestyle="dotted",
                    color="black",
                    linewidth=1.0,
                )

            fig.suptitle(
                f"{ds_train.name_human()}"
                "\n"
                f"({m.name_network})"
                "  "
                f"Resample-freq: {n_epochs_stale_max} + Max-resample: {n_remasks_stale_max}"
            )
            self._style_y_as_error(ax, 100 * y_max)
            ax.set_ylabel("error [MSE%]")

            ax.legend()

            fig.tight_layout()
            fig.savefig(f"{path}.png")
            plt.close(fig)

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

    def island_ring(self) -> None:
        mask_train = util_dataset.MaskerIsland(0.5)

        ds_train = self._dataset_single(mask_train)
        ds_train.as_train(self._n_instances_train)
        for m in list(self._models_current_single(ds_train)):
            ds_eval = self._dataset_single(ds_train.masks[0])
            ds_eval.as_eval(self._n_instances_eval)
            m.datasets_eval = [ds_eval]
            m.load_network()

            m.datasets_eval = self._datasets_evals_single_random
            m.eval(print_result=True)
            m.datasets_eval = self._datasets_evals_single_island
            m.eval(print_result=True)
            m.datasets_eval = self._datasets_evals_single_ring
            m.eval(print_result=True)

    def mask_nonstandard(self) -> None:
        for value_mask in [0.5, 0.0, 1.0]:  # include default 0.5 for direct comparison
            mask_train = util_dataset.MaskerRandom.from_min_max(
                intensity_min=0.0, intensity_max=1.0, value_mask=value_mask
            )
            ds_train = self._dataset_single(mask_train)
            ds_train.as_train(self._n_instances_train)

            for m in [list(self._models_current_single(ds_train))[0]]:  # fno only
                ds_eval = self._dataset_single(ds_train.masks[0])
                ds_eval.as_eval(self._n_instances_eval)
                m.datasets_eval = [ds_eval]
                m.load_network()

                m.datasets_eval = self._datasets_evals_single_random
                m.eval(print_result=True)
                m.datasets_eval = self._datasets_evals_single_island
                m.eval(print_result=True)
                m.datasets_eval = self._datasets_evals_single_ring
                m.eval(print_result=True)

    def plot_error_single(self) -> None:
        y_max, y_clip = 0.18, 0.175
        for ds_train in self._datasets_train_single:
            models = list(self._models_current_single(ds_train))
            fig, axs = plt.subplots(1, 3, figsize=(13, 5), dpi=200)
            for m in models:
                m.load_network()
                for ax, ds_eval in zip(
                    axs,
                    [
                        self._datasets_evals_single_random,
                        self._datasets_evals_single_island,
                        self._datasets_evals_single_ring,
                    ],
                ):
                    m.datasets_eval = ds_eval
                    self._plot_errors(
                        ax, m.errors(clip_at_max=y_clip), label=m.name_network
                    )

            for ax, mask_name in zip(axs, ["Random", "Island", "Ring"]):
                self._style_plot(
                    ax,
                    mask_min_max=ds_train.masks[0].min_max(as_percentage=True),
                    y_max=y_max * 100,
                    y_ticks=[2.5, 5, 10, 15, 17.5],
                )
                ax.set_title(f"(Eval-)Masking: {mask_name}")
            axs[0].set_ylabel("MSE%")

            fig.suptitle(ds_train.name_human(multiline=True))
            fig.tight_layout()

            path = pathlib.Path(f"{ds_train.path}.png")
            fig.savefig(path)
            plt.close(fig)

    def plot_error_double(self) -> None:
        y_max, y_clip = 0.26, 0.25
        for ds_train in self._datasets_train_double:
            models = list(self._models_current_double(ds_train))
            fig, axs_channels = plt.subplots(2, 3, figsize=(13, 9), dpi=200)
            for m in models:
                m.load_network()

                for i, ds_eval in enumerate(
                    [
                        self._datasets_evals_double_random,
                        self._datasets_evals_double_island,
                        self._datasets_evals_double_ring,
                    ]
                ):
                    m.datasets_eval = ds_eval
                    errors_chan_0, errors_chan_1 = m.errors(clip_at_max=y_clip)
                    self._plot_errors(
                        axs_channels[0][i], errors_chan_0, label=m.name_network
                    )
                    self._plot_errors(
                        axs_channels[1][i], errors_chan_1, label=m.name_network
                    )

            for idx_channel, axs in enumerate(axs_channels):
                for ax, mask_name in zip(axs, ["Random", "Island", "Ring"]):
                    self._style_plot(
                        ax,
                        mask_min_max=ds_train.masks[0].min_max(as_percentage=True),
                        y_max=y_max * 100,
                        y_ticks=[i * 5 for i in range(6)],
                        baseline_high=20.0,
                    )
                    ax.set_title(f"Channel {idx_channel} Masking: {mask_name}")
                axs[0].set_ylabel("MSE%")

            fig.suptitle(ds_train.name_human(multiline=True))
            fig.tight_layout()

            path = pathlib.Path(f"{ds_train.path}.png")
            fig.savefig(path)
            plt.close(fig)

    def _plot_errors(
        self,
        ax: mpl.axes.Axes,
        errors: np.ndarray,
        label: str,
    ) -> None:
        style = {"linestyle": "dashed", "linewidth": 1.0, "marker": "x"}
        ax.plot(
            self._intensities_eval,
            errors,
            **style,
            label=label,
        )

    def _style_plot(
        self,
        ax: mpl.axes.Axes,
        mask_min_max: tuple[float, float],
        y_max: float,
        y_ticks: typing.Sequence[float],
        baseline_low: float = 5.0,
        baseline_high: float = 15.0,
    ) -> None:
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

        self._style_y_as_error(ax, y_max, y_ticks, baseline_low, baseline_high)

    def _style_y_as_error(
        self,
        ax: mpl.axes.Axes,
        y_max: float,
        y_ticks: typing.Optional[typing.Sequence[float]] = None,
        baseline_low: float = 5.0,
        baseline_high: float = 15.0,
    ) -> None:
        ax.set_ylim(0.0, y_max)
        if y_ticks:
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

    def plot_reconstruction(self) -> None:
        for ds_eval, name in zip(
            [
                self._datasets_evals_single_random[5],
                self._datasets_evals_single_island[5],
                self._datasets_evals_single_ring[5],
            ],
            ["random", "island", "ring"],
        ):
            ds_train = self._datasets_train_single[0]
            m = next(self._models_current_single(ds_train))
            m.load_network()
            m.datasets_eval = [ds_eval]
            ((truth, masked, ours, error),), mse = m.reconstruct()

            pt = util_plot.PlotIllustration(
                ds_eval.grids,
                ticks_x1_3d=[0.0, 0.20, 0.40, 0.60],
                ticks_x2_3d=[0.0, 0.20, 0.40, 0.60],
            )
            pt.make_fig_ax(4)

            pt.plot_targets_uniform(
                [truth, masked, ours],
                ["Truth", "Masked", "Prediction"],
                _min=0.05,
                _max=0.95,
            )

            pt.plot_2d(
                pt.axs_2d[3],
                error,
                title=f"Raw Difference (MSE: {mse:.2%})",
                _min=-0.2,
                _max=+0.2,
                colormap="plasma",
            )
            pt.plot_3d(pt.axs_3d[3], error, _min=-0.7, _max=+0.7, colormap="plasma")

            title = (
                f"Training> {ds_train.name_human(multiline=False)}"
                "\n"
                f"Evaluation> {ds_eval.name_human(multiline=False)}"
            )
            pt.finalize(
                pathlib.Path(f"{ds_train.path}--reconstruction-single-{name}.png"),
                title=title,
            )

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

    def _dataset_raw(self) -> dataset_poisson.DatasetPoisson2d:
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

    def plot_remask(self) -> None:
        self._plot_train_log("full", "unet")

    def plot_raw(self) -> None:
        DEFINITION.seed(42)
        ds_raw = self._dataset_raw()
        ds_raw.plot_uf()

        DEFINITION.seed(42)
        dataset_poisson.DatasetGauss(
            grid.Grids(
                [
                    grid.Grid(n_pts=64, stepsize=0.01, start=0.0),
                    grid.Grid(n_pts=64, stepsize=0.01, start=0.0),
                ]
            )
        ).plot_uf()


class ProblemHeat(Problem):
    def __init__(self):
        super().__init__(n_channels_raw=dataset_heat.DatasetHeat.N_CHANNELS)

    def _dataset_raw(self) -> dataset_heat.DatasetHeat:
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

    def plot_raw(self) -> None:
        ds_raw = self._dataset_raw()
        ds_raw.plot_snapshots()


class ProblemWave(Problem):
    def __init__(self):
        super().__init__(n_channels_raw=dataset_wave.DatasetWave.N_CHANNELS)

    def _dataset_raw(self) -> dataset_wave.DatasetWave:
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

    def plot_remask(self) -> None:
        self._plot_train_log("full", "kno")

    def plot_raw(self) -> None:
        ds_raw = self._dataset_raw()
        ds_raw.plot_snapshots()


class ProblemMask:
    def __init__(self):
        self._mask_random = util_dataset.MaskerRandom(0.5, intensity_spread=0.0)
        self._mask_island = util_dataset.MaskerIsland(0.75, intensity_spread=0.0)
        self._mask_ring = util_dataset.MaskerRing(0.25, intensity_spread=0.0)

    def plot(self) -> None:
        self._mask_random.plot(resolution=25)
        self._mask_island.plot(resolution=100)
        self._mask_ring.plot(resolution=100)
