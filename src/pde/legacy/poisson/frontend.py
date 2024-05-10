import abc
import collections
import logging
import pathlib
import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.deepl import cno, fno_2d
from src.numerics import grid
from src.pde.dataset import DatasetMasked, DatasetSplits
from src.pde.legacy.poisson.learner import (
    LearnerPoissonCNOMaskedSolution,
    LearnerPoissonCNOMaskedSolutionSource,
    LearnerPoissonFNOMaskedSolution,
    LearnerPoissonFNOMaskedSolutionSource,
    LearnerPoissonFourier,
)
from src.pde.poisson.dataset import (
    DatasetGauss,
    DatasetPoisson2d,
    DatasetPoissonMaskedSolution,
    DatasetPoissonMaskedSolutionSource,
    DatasetSin,
)
from src.util.dataset import Masker, MaskerIsland, MaskerRandom
from src.util.saveload import SaveloadImage, SaveloadTorch

logger = logging.getLogger(__name__)


class PlotFormat:
    def __init__(
        self,
        plot_min: float = 0.0,
        plot_max: float = 110.0,
        clip_min: float = 0.0,
        clip_max: float = 105.0,
        baseline_low: float = 20.0,
        baseline_high: float = 100.0,
    ):
        self._plot_min, self._plot_max = plot_min, plot_max
        self._clip_min, self._clip_max = clip_min, clip_max
        self._baseline_low, self._baseline_high = baseline_low, baseline_high

    def format_y_axis(self, ax: mpl.axes.Axes, use_percent: bool = True) -> None:
        ax.set_ylim(self._plot_min, self._plot_max)
        ax.axhline(
            self._baseline_low, linestyle="dashed", color="lightgrey", linewidth=1.5
        )
        ax.axhline(
            self._baseline_high, linestyle="dashed", color="darkgrey", linewidth=1.5
        )

        if use_percent:
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=1))

    def clip(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values, a_min=self._clip_min, a_max=self._clip_max)


class DatasetsMasked:
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        ds_raw: DatasetPoisson2d,
        saveload: SaveloadTorch,
    ):
        self._grid_x1, self._grid_x2 = grid_x1, grid_x2

        self._ds_raw = ds_raw
        self._saveload = saveload
        self._ds_eval_raw, self._ds_train_raw = self._make_raw()

        self._percs = np.arange(start=0.1, stop=1.0, step=0.1)

    def _make_raw(
        self,
    ) -> tuple[
        torch.utils.data.dataset.TensorDataset, torch.utils.data.dataset.TensorDataset
    ]:
        n_eval, n_train = 300, 1500
        location_eval, location_train = (
            self._saveload.rebase_location(f"dataset--eval-{n_eval}"),
            self._saveload.rebase_location(f"dataset--train-{n_train}"),
        )
        if not (location_eval.exists() and location_train.exists()):
            n_raw = 3000
            location_raw = self._saveload.rebase_location(f"dataset--raw-{n_raw}")
            if not self._saveload.exists(location_raw):
                self._saveload.save(
                    self._ds_raw.as_dataset(n_raw),
                    location_raw,
                )
            ds_eval, ds_train = DatasetSplits(self._saveload.load(location_raw)).split(
                n_eval, n_train
            )
            self._saveload.save(ds_eval, location_eval, overwrite=True)
            self._saveload.save(ds_train, location_train, overwrite=True)

        return (
            self._saveload.load(location_eval),
            self._saveload.load(location_train),
        )

    def datasets_train(self) -> typing.Generator[DatasetMasked, None, None]:
        for perc in self._percs:
            yield DatasetPoissonMaskedSolution(
                self._grid_x1,
                self._grid_x2,
                self._ds_train_raw,
                MaskerRandom(perc_to_mask=perc),
            )

    def datasets_eval_random(self) -> typing.Generator[DatasetMasked, None, None]:
        for perc in self._percs:
            yield DatasetPoissonMaskedSolution(
                self._grid_x1,
                self._grid_x2,
                self._ds_eval_raw,
                MaskerRandom(perc_to_mask=perc),
            )
            yield DatasetPoissonMaskedSolution(
                self._grid_x1,
                self._grid_x2,
                self._ds_eval_raw,
                MaskerIsland(perc_to_keep=1 - perc),
            )

    def datasets_eval_island(self) -> typing.Generator[DatasetMasked, None, None]:
        for perc in self._percs:
            yield DatasetPoissonMaskedSolution(
                self._grid_x1,
                self._grid_x2,
                self._ds_eval_raw,
                MaskerIsland(perc_to_keep=1 - perc),
            )

    def plot_raw(self, saveas: str) -> None:
        fig = self._ds_raw.plot(set_title_upper=False)
        fig.savefig(saveas)
        plt.close(fig)

    def plot_masked(self) -> None:
        # NOTE:
        #   for llustration in thesis
        ds = DatasetPoissonMaskedSolution(
            self._grid_x1,
            self._grid_x2,
            self._ds_eval_raw,
            MaskerRandom(0.3),
        )
        fig = ds.plot_instance(n_instances=3)
        fig.savefig("random")
        plt.close(fig)

        ds = DatasetPoissonMaskedSolution(
            self._grid_x1,
            self._grid_x2,
            self._ds_eval_raw,
            MaskerIsland(0.7),
        )
        fig = ds.plot_instance(n_instances=3)
        fig.savefig("island")
        plt.close(fig)


class Pipeline:
    def __init__(
        self,
        dataset_raw: DatasetPoisson2d,
        network_name: str,
    ):
        self._grid_x1 = grid.Grid(n_pts=64, stepsize=0.01, start=0.0)
        self._grid_x2 = grid.Grid(n_pts=64, stepsize=0.01, start=0.0)
        self._grids = grid.Grids([self._grid_x1, self._grid_x2])

        self._dataset_raw = dataset_raw
        self._saveload_base = f"poisson/{self._dataset_raw.as_name()}"
        self._saveload = SaveloadTorch(self._saveload_base)

        self._network_name = network_name

        self._percs = np.arange(start=0.1, stop=1.0, step=0.1)
        self._ds_eval_raw, self._ds_train_raw = self._dataset_splits()

    def _name_model(self, name_dataset: str) -> pathlib.Path:
        return self._saveload.rebase_location(
            f"network--{self._network_name}--{name_dataset}"
        )

    def _dataset_splits(
        self,
    ) -> tuple[
        torch.utils.data.dataset.TensorDataset, torch.utils.data.dataset.TensorDataset
    ]:
        n_eval, n_train = 300, 1500
        location_eval, location_train = (
            self._saveload.rebase_location(f"dataset--eval-{n_eval}"),
            self._saveload.rebase_location(f"dataset--train-{n_train}"),
        )
        if not (location_eval.exists() and location_train.exists()):
            n_raw = 3000
            location_raw = self._saveload.rebase_location(f"dataset--raw-{n_raw}")
            if not self._saveload.exists(location_raw):
                self._saveload.save(
                    self._dataset_raw.as_dataset(n_raw),
                    location_raw,
                )
            ds_eval, ds_train = DatasetSplits(self._saveload.load(location_raw)).split(
                n_eval, n_train
            )
            self._saveload.save(ds_eval, location_eval, overwrite=True)
            self._saveload.save(ds_train, location_train, overwrite=True)

        return (
            self._saveload.load(location_eval),
            self._saveload.load(location_train),
        )

    @abc.abstractmethod
    def models(
        self,
    ) -> collections.abc.Generator[
        tuple[DatasetMasked, LearnerPoissonFourier], None, None
    ]:
        pass

    def _model_trained(
        self, dataset: DatasetMasked, n_epochs: int = 1001, batch_size: int = 20
    ) -> LearnerPoissonFourier:
        location = self._name_model(dataset.as_name())
        if self._saveload.exists(location):
            learner = self._load_learner_trained(location)
        else:
            network = self._make_network()
            learner = self._load_learner_untrained(network)
            logger.info(
                "training model with ["
                f"{sum(p.numel() for p in network.parameters() if p.requires_grad)}"
                f"] parameters; to be saved at [{location}]"
            )
            learner.train(
                dataset.make(),
                n_epochs=n_epochs,
                batch_size=batch_size,
                dataset_eval=self._dataset_eval_during_train(),
            )
            self._saveload.save(network, location)
        return learner

    @abc.abstractmethod
    def _load_learner_trained(self, location: pathlib.Path) -> LearnerPoissonFourier:
        raise NotImplementedError

    @abc.abstractmethod
    def _make_network(self) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def _load_learner_untrained(
        self, network: torch.nn.Module
    ) -> LearnerPoissonFourier:
        raise NotImplementedError

    @abc.abstractmethod
    def _dataset_eval_during_train(self) -> torch.utils.data.dataset.TensorDataset:
        raise NotImplementedError

    def train(self) -> None:
        for __ in self.models():
            pass

    def _plot_compare(
        self,
        learner: LearnerPoissonFourier,
        dataset_train: DatasetMasked,
        dataset_eval: DatasetMasked,
    ) -> None:
        ds = dataset_eval.make()
        saveload = SaveloadImage(self._saveload_base)
        location = f"plot_compare--{dataset_train.as_name()}--{dataset_eval.as_name()}"
        saveload.save(
            learner.plot_comparison_2d(ds),
            saveload.rebase_location(f"{location}--2d"),
            overwrite=False,
        )
        saveload.save(
            learner.plot_comparison_3d(ds),
            saveload.rebase_location(f"{location}--3d"),
            overwrite=False,
        )


class PipelineMaskedSolution(Pipeline):
    def models(
        self, include_random_masks: bool = True, include_island_masks: bool = True
    ) -> collections.abc.Generator[
        tuple[DatasetMasked, LearnerPoissonFourier], None, None
    ]:
        for perc in self._percs:
            if include_random_masks:
                dataset = DatasetPoissonMaskedSolution(
                    self._grid_x1,
                    self._grid_x2,
                    self._ds_train_raw,
                    MaskerRandom(perc_to_mask=perc),
                )
                yield dataset, self._model_trained(dataset)
            if include_island_masks:
                dataset = DatasetPoissonMaskedSolution(
                    self._grid_x1,
                    self._grid_x2,
                    self._ds_train_raw,
                    MaskerIsland(perc_to_keep=1 - perc),
                )
                yield dataset, self._model_trained(dataset)

    def datasets_masked(
        self, from_eval: bool = True, random_style: bool = True
    ) -> typing.Generator[DatasetMasked, None, None]:
        for perc in np.arange(start=0.1, stop=1.0, step=0.1):
            dataset_raw = self._ds_eval_raw if from_eval else self._ds_train_raw
            mask = (
                MaskerRandom(perc_to_mask=perc)
                if random_style
                else MaskerIsland(perc_to_keep=1 - perc)
            )
            yield DatasetPoissonMaskedSolution(
                self._grid_x1, self._grid_x2, dataset_raw, mask
            )

    def _dataset_eval_during_train(self) -> torch.utils.data.dataset.TensorDataset:
        return DatasetPoissonMaskedSolution(
            self._grid_x1, self._grid_x2, self._ds_eval_raw, MaskerRandom(0.5)
        ).make()

    def print_result_eval(self) -> None:
        for dataset_train, learner in self.models():
            print(f"eval> model trained on {dataset_train.as_name()}")
            for dataset_eval in self.datasets_masked(from_eval=False):
                print(f"eval> dataset {dataset_eval.as_name()}")
                learner.eval(dataset_eval.make(), print_result=True)
                print()
            print()

    def plot_model_2d_3d(self) -> None:
        for dataset_train, learner in self.models():
            for dataset_eval in self.datasets_masked():
                self._plot_compare(learner, dataset_train, dataset_eval)

    def plot_models_all(self) -> None:
        saveload = SaveloadImage(self._saveload_base)
        location = saveload.rebase_location(f"mask_one--{self._network_name}")
        if saveload.exists(location):
            logger.info(f"plot already done [{location}],  skipping")
            return

        fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(15, 5), dpi=200)
        fig.suptitle(
            f"Model: {self._network_name.upper()}"
            "\n"
            f"Dataset: {self._dataset_raw.as_name()}"
        )
        self._plot_models_all_mask_one(ax_1, perc_random=0.1, perc_island=0.9)
        self._plot_models_all_mask_one(ax_2, perc_random=0.5, perc_island=0.5)
        self._plot_models_all_mask_one(ax_3, perc_random=0.9, perc_island=0.1)

        saveload.save(fig, location, overwrite=True)
        plt.close(fig)

    def _plot_models_all_mask_one(
        self,
        ax: mpl.axes.Axes,
        perc_random: float = 0.5,
        perc_island: float = 0.5,
    ) -> None:
        mask_random, mask_island = MaskerRandom(perc_random), MaskerIsland(perc_island)
        errors_t_random_e_random, errors_t_random_e_island = self._errors(
            random_style=True, mask_random=mask_random, mask_island=mask_island
        )
        errors_t_island_e_random, errors_t_island_e_island = self._errors(
            random_style=False, mask_random=mask_random, mask_island=mask_island
        )

        style_line = {"linewidth": 1.0, "linestyle": "dashed"}
        ax.plot(
            self._percs,
            errors_t_random_e_random,
            **style_line,
            marker="+",
            label=f"train: random(*); eval: {mask_random.name()}",
        )
        ax.plot(
            self._percs,
            errors_t_random_e_island,
            **style_line,
            marker="+",
            label=f"train: random(*); eval: {mask_island.name()}",
        )
        ax.plot(
            self._percs,
            errors_t_island_e_random,
            **style_line,
            marker="x",
            label=f"train: island(*); eval: {mask_random.name()}",
        )
        ax.plot(
            self._percs,
            errors_t_island_e_island,
            **style_line,
            marker="x",
            label=f"train: island(*); eval: eval: {mask_island.name()}",
        )
        ax.set_xlabel("masking intensity")
        ax.set_ylabel("error [$L^2$]")
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=1))

        PlotFormat().format_y_axis(ax)
        ax.legend()

    def _errors(
        self, random_style: bool, mask_random: Masker, mask_island: Masker
    ) -> tuple[list[float], list[float]]:
        errors_t_random_e_random, errors_t_random_e_island = [], []
        for __, learner in self.models(
            include_random_masks=random_style, include_island_masks=not random_style
        ):
            # errors_t_random_e_random.append(10 * np.random.rand() + 30)
            # errors_t_random_e_island.append(10 * np.random.rand() + 30)
            errors_t_random_e_random.append(
                learner.eval(
                    DatasetPoissonMaskedSolution(
                        self._grid_x1, self._grid_x2, self._ds_eval_raw, mask_random
                    ).make()
                )
                * 100  # make percentage
            )
            errors_t_random_e_island.append(
                learner.eval(
                    DatasetPoissonMaskedSolution(
                        self._grid_x1, self._grid_x2, self._ds_eval_raw, mask_island
                    ).make()
                )
                * 100  # make percentage
            )
        return errors_t_random_e_random, errors_t_random_e_island

    def plot_model_one_masks_all(self) -> None:
        for dataset_train, learner in self.models():
            self._plot_model_one_masks_all(learner, dataset_train)

    def _plot_model_one_masks_all(
        self,
        learner: LearnerPoissonFourier,
        dataset_train: DatasetMasked,
    ) -> None:
        saveload = SaveloadImage(self._saveload_base)
        location = saveload.rebase_location(
            f"all_masks--{self._network_name}--{dataset_train.as_name()}"
        )
        if saveload.exists(location):
            logger.info(f"plot already done [{location}],  skipping")
        else:
            fig, ax = plt.subplots(dpi=200)
            style = {"linestyle": "dashed", "linewidth": 1.5, "marker": "x"}
            ax.plot(
                self._percs,
                self._errors_eval(learner, random_style=True),
                **style,
                label="random masking",
            )
            ax.plot(
                self._percs,
                self._errors_eval(learner, random_style=False),
                **style,
                label="island masking",
            )
            ax.set_xlabel("masking intensity")
            ax.set_ylabel("error [$L^2$]")
            ax.set_title(
                f"Model: {self._network_name.upper()}\n"
                f"Dataset: {self._dataset_raw.as_name()}; "
                f"Mask: {dataset_train.as_name()}"
            )
            PlotFormat().format_y_axis(ax)
            ax.legend()

            saveload.save(fig, location, overwrite=True)
            plt.close(fig)

    def _errors_eval(
        self, learner: LearnerPoissonFourier, random_style: bool = True
    ) -> typing.Sequence[float]:
        errors = []
        for ds in self.datasets_masked(random_style=random_style):
            errors.append(learner.eval(ds.make()) * 100)
        return errors


class PipelineFNOMaskedSolution(PipelineMaskedSolution):
    def __init__(self, dataset_raw: DatasetPoisson2d):
        super().__init__(network_name="fno", dataset_raw=dataset_raw)

    def _load_learner_trained(self, location: pathlib.Path) -> LearnerPoissonFourier:
        return LearnerPoissonFNOMaskedSolution(
            self._grid_x1,
            self._grid_x2,
            self._saveload.load(location),
        )

    def _make_network(self) -> torch.nn.Module:
        return fno_2d.FNO2d(n_channels_lhs=4, n_channels_rhs=1)

    def _load_learner_untrained(
        self, network: torch.nn.Module
    ) -> LearnerPoissonFourier:
        return LearnerPoissonFNOMaskedSolution(
            self._grid_x1,
            self._grid_x2,
            network,
        )


class PipelineCNOMaskedSolution(PipelineMaskedSolution):
    def __init__(self, dataset_raw: DatasetPoisson2d):
        super().__init__(network_name="cno", dataset_raw=dataset_raw)

    def _load_learner_trained(self, location: pathlib.Path) -> LearnerPoissonFourier:
        return LearnerPoissonCNOMaskedSolution(
            self._grid_x1,
            self._grid_x2,
            self._saveload.load(location),
        )

    def _make_network(self) -> torch.nn.Module:
        return cno.CNO2d(in_channel=4, out_channel=1)

    def _load_learner_untrained(
        self, network: torch.nn.Module
    ) -> LearnerPoissonFourier:
        return LearnerPoissonCNOMaskedSolution(
            self._grid_x1,
            self._grid_x2,
            network,
        )


class PipelineMaskedSolutionSource(Pipeline):
    def models(
        self,
    ) -> collections.abc.Generator[
        tuple[DatasetPoissonMaskedSolutionSource, LearnerPoissonFourier],
        None,
        None,
    ]:
        for perc in self._percs:
            mask_solution = MaskerRandom(perc_to_mask=perc)
            mask_source = MaskerRandom(perc_to_mask=perc)

            dataset = DatasetPoissonMaskedSolutionSource(
                self._grid_x1,
                self._grid_x2,
                self._ds_train_raw,
                mask_solution=mask_solution,
                mask_source=mask_source,
            )
            yield dataset, self._model_trained(dataset)

    def _dataset_eval_during_train(self) -> torch.utils.data.dataset.TensorDataset:
        return DatasetPoissonMaskedSolutionSource(
            self._grid_x1,
            self._grid_x2,
            self._ds_eval_raw,
            MaskerRandom(0.5),
            MaskerRandom(0.5),
        ).make()

    def print_result_eval(self) -> None:
        for dataset_train, learner in self.models():
            for dataset_eval in self.datasets_masked():
                print(
                    "eval> [train, test]: "
                    f"[{dataset_train.as_name()}, {dataset_eval.as_name()}]"
                )
                learner.eval(dataset_eval.make(), print_result=True)
                print()

    def datasets_masked(
        self, from_eval: bool = True, random_style: bool = True
    ) -> typing.Generator[DatasetMasked, None, None]:
        for perc in np.arange(start=0.1, stop=1.0, step=0.1):
            dataset_raw = self._ds_eval_raw if from_eval else self._ds_train_raw
            mask = (
                MaskerRandom(perc_to_mask=perc)
                if random_style
                else MaskerIsland(perc_to_keep=1 - perc)
            )
            yield DatasetPoissonMaskedSolutionSource(
                self._grid_x1, self._grid_x2, dataset_raw, mask, mask
            )

    def plot_model_one_masks_all(self) -> None:
        for dataset_train, learner in self.models():
            self._plot_model_one_masks_all(learner, dataset_train)

    def _plot_model_one_masks_all(
        self,
        learner: LearnerPoissonFourier,
        dataset_train: DatasetMasked,
    ) -> None:
        saveload = SaveloadImage(self._saveload_base)
        location = saveload.rebase_location(
            f"all_masks--{self._network_name}--{dataset_train.as_name()}"
        )
        if saveload.exists(location):
            logger.info(f"plot already done [{location}],  skipping")
        else:
            fig, ax = plt.subplots(dpi=200)
            style_solution = {"linestyle": "dashed", "linewidth": 1.5, "marker": "x"}
            style_source = {"linestyle": "dashed", "linewidth": 1.5, "marker": "+"}

            errors_random_solution, errors_random_source = self._errors_eval(
                learner, random_style=True
            )
            errors_island_solution, errors_island_source = self._errors_eval(
                learner, random_style=False
            )

            ax.plot(
                self._percs,
                errors_random_solution,
                **style_solution,
                label="solution $u$: random masking",
            )
            ax.plot(
                self._percs,
                errors_island_solution,
                **style_solution,
                label="solution $u$: island masking",
            )
            ax.plot(
                self._percs,
                errors_random_source,
                **style_source,
                label="source $f$: random masking",
            )
            ax.plot(
                self._percs,
                errors_island_source,
                **style_source,
                label="source $f$: island masking",
            )

            ax.set_xlabel("masking intensity")
            ax.set_ylabel("error [$L^2$]")
            ax.set_title(
                f"Model: {self._network_name.upper()}\n"
                f"Dataset: {self._dataset_raw.as_name()}; "
                f"Mask: {dataset_train.as_name()}"
            )
            ax.axhline(20.0, linestyle="dashed", color="lightgrey", linewidth=1.5)
            self._set_y_limit(
                ax,
                np.array(
                    [
                        errors_random_solution,
                        errors_random_source,
                        errors_island_solution,
                        errors_island_source,
                    ]
                ),
            )
            ax.legend()

            saveload.save(fig, location, overwrite=True)
            plt.close(fig)

    @staticmethod
    def _set_y_limit(
        ax: mpl.axes.Axes,
        values: np.ndarray,
        limit: float = 100.0,
    ) -> None:
        if np.max(values) > limit:
            ax.set_ylim(0, 110.0)
            ax.axhline(limit, linestyle="dashed", color="darkgrey", linewidth=1.5)

    def _errors_eval(
        self, learner: LearnerPoissonFourier, random_style: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        errors_solution, errors_source = [], []
        for ds in self.datasets_masked(random_style=random_style):
            errors = learner.errors(ds.make())
            errors_solution.append(errors[0] * 100)
            errors_source.append(errors[1] * 100)
        return (
            np.clip(np.array(errors_solution), a_min=None, a_max=105.0),
            np.clip(np.array(errors_source), a_min=None, a_max=105.0),
        )


class PipelineFNOMaskedSolutionSource(PipelineMaskedSolutionSource):
    def __init__(self, dataset_raw: DatasetPoisson2d):
        super().__init__(network_name="fno", dataset_raw=dataset_raw)

    def _load_learner_trained(self, location: pathlib.Path) -> LearnerPoissonFourier:
        return LearnerPoissonFNOMaskedSolutionSource(
            self._grid_x1,
            self._grid_x2,
            self._saveload.load(location),
        )

    def _make_network(self) -> torch.nn.Module:
        return fno_2d.FNO2d(n_channels_lhs=4, n_channels_rhs=2)

    def _load_learner_untrained(
        self, network: torch.nn.Module
    ) -> LearnerPoissonFourier:
        return LearnerPoissonFNOMaskedSolutionSource(
            self._grid_x1,
            self._grid_x2,
            network,
        )


class PipelineCNOMaskedSolutionSource(PipelineMaskedSolutionSource):
    def __init__(self, dataset_raw: DatasetPoisson2d):
        super().__init__(network_name="cno", dataset_raw=dataset_raw)

    def _load_learner_trained(self, location: pathlib.Path) -> LearnerPoissonFourier:
        return LearnerPoissonCNOMaskedSolutionSource(
            self._grid_x1,
            self._grid_x2,
            self._saveload.load(location),
        )

    def _make_network(self) -> torch.nn.Module:
        return cno.CNO2d(in_channel=4, out_channel=2)

    def _load_learner_untrained(
        self, network: torch.nn.Module
    ) -> LearnerPoissonFourier:
        return LearnerPoissonCNOMaskedSolutionSource(
            self._grid_x1,
            self._grid_x2,
            network,
        )


class Main:
    def __init__(self):
        self._grid_x1 = grid.Grid(n_pts=64, stepsize=0.01, start=0.0)
        self._grid_x2 = grid.Grid(n_pts=64, stepsize=0.01, start=0.0)

    def plot_raws(self) -> None:
        for ds_raw in [
            DatasetSin(self._grid_x1, self._grid_x2),
            DatasetGauss(self._grid_x1, self._grid_x2),
        ]:
            saveload = SaveloadTorch(f"poisson/{ds_raw.as_name()}")

            DatasetsMasked(self._grid_x1, self._grid_x2, ds_raw, saveload).plot_raw(
                ds_raw.as_name()
            )

    def plot_masks(self) -> None:
        ds_raw = DatasetSin(self._grid_x1, self._grid_x2)
        saveload = SaveloadTorch(f"poisson/{ds_raw.as_name()}")
        DatasetsMasked(self._grid_x1, self._grid_x2, ds_raw, saveload).plot_masked()

    def plot(self) -> None:
        for ds_raw in [
            DatasetSin(self._grid_x1, self._grid_x2),
            DatasetGauss(self._grid_x1, self._grid_x2),
        ]:
            fno_m = PipelineFNOMaskedSolution(ds_raw)
            fno_m.train()
            fno_m.plot_models_all()

            cno_m = PipelineCNOMaskedSolution(ds_raw)
            cno_m.train()
            cno_m.plot_models_all()

    def mask_one(self) -> None:
        for ds_raw in [
            DatasetGauss(self._grid_x1, self._grid_x2),
            DatasetSin(self._grid_x1, self._grid_x2),
        ]:
            for pipe in [
                PipelineFNOMaskedSolution(ds_raw),
                PipelineCNOMaskedSolution(ds_raw),
            ]:
                pipe.plot_model_one_masks_all()

    def mask_double(self) -> None:
        for ds_raw in [
            DatasetGauss(self._grid_x1, self._grid_x2),
            DatasetSin(self._grid_x1, self._grid_x2),
        ]:
            for pipe in [
                PipelineFNOMaskedSolutionSource(ds_raw),
                PipelineCNOMaskedSolutionSource(ds_raw),
            ]:
                pipe.plot_model_one_masks_all()


def main() -> None:
    m = Main()
    m.mask_one()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )

    torch.manual_seed(42)
    main()
