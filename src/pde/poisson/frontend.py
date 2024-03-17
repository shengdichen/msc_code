import abc
import collections
import logging
import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.deepl import cno, fno_2d
from src.numerics import grid
from src.pde.dataset import DatasetMasked, DatasetSplits
from src.pde.poisson.dataset import (DatasetConstructed, DatasetConstructedSin,
                                     DatasetGauss,
                                     DatasetPoissonMaskedSolution,
                                     DatasetPoissonMaskedSolutionSource,
                                     DatasetSumOfGauss)
from src.pde.poisson.learner import (LearnerPoissonCNOMaskedSolution,
                                     LearnerPoissonCNOMaskedSolutionSource,
                                     LearnerPoissonFNOMaskedSolution,
                                     LearnerPoissonFNOMaskedSolutionSource,
                                     LearnerPoissonFourier)
from src.util.dataset import Masker, MaskerIsland, MaskerRandom
from src.util.saveload import SaveloadImage, SaveloadTorch

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, network_name: str):
        self._grid_x1 = grid.Grid(n_pts=64, stepsize=0.01, start=0.0)
        self._grid_x2 = grid.Grid(n_pts=64, stepsize=0.01, start=0.0)
        self._grids = grid.Grids([self._grid_x1, self._grid_x2])

        self._dataset_name, self._network_name = "sum_of_gauss", network_name
        self._saveload_base = f"poisson/{self._dataset_name}"
        self._saveload = SaveloadTorch(self._saveload_base)

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
        n_eval, n_train = 2500, 500
        location_eval, location_train = (
            self._saveload.rebase_location(f"dataset--eval-{n_eval}"),
            self._saveload.rebase_location(f"dataset--train-{n_train}"),
        )
        if not (location_eval.exists() and location_train.exists()):
            n_raw = 3000
            location_raw = self._saveload.rebase_location(f"dataset--raw-{n_raw}")
            if not self._saveload.exists(location_raw):
                self._saveload.save(
                    DatasetGauss(self._grid_x1, self._grid_x2).as_dataset(n_raw),
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

    @abc.abstractmethod
    def _model_trained(self, dataset: DatasetMasked) -> LearnerPoissonFourier:
        pass

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

    def plot_models_all_mask_one(
        self, perc_random: float = 0.5, perc_island: float = 0.5
    ) -> None:
        saveload = SaveloadImage(self._saveload_base)
        location = saveload.rebase_location(f"mask_one--{self._network_name}")
        if saveload.exists(location):
            logger.info(f"plot already done [{location}],  skipping")
            return

        mask_random, mask_island = MaskerRandom(perc_random), MaskerIsland(perc_island)
        errors_t_random_e_random, errors_t_random_e_island = self._errors(
            random_style=True, mask_random=mask_random, mask_island=mask_island
        )
        errors_t_island_e_random, errors_t_island_e_island = self._errors(
            random_style=False, mask_random=mask_random, mask_island=mask_island
        )

        fig, ax = plt.subplots(dpi=200)
        style = {"linestyle": "dashed", "marker": "x"}
        ax.plot(
            self._percs,
            errors_t_random_e_random,
            **style,
            label=f"train: random(*); eval: {mask_random.as_name()}",
        )
        ax.plot(
            self._percs,
            errors_t_random_e_island,
            **style,
            label=f"train: random(*); eval: {mask_island.as_name()}",
        )
        ax.plot(
            self._percs,
            errors_t_island_e_random,
            **style,
            label=f"train: island(*); eval: {mask_random.as_name()}",
        )
        ax.plot(
            self._percs,
            errors_t_island_e_island,
            **style,
            label=f"train: island(*); eval: eval: {mask_island.as_name()}",
        )
        ax.set_xlabel("masking proportion")
        ax.set_ylabel("error [$L^2$]")
        ax.set_title(
            "error VS masking \n"
            f"{self._network_name} on "
            f'["{self._dataset_name}"]'
        )
        ax.legend()

        saveload.save(fig, location, overwrite=True)
        plt.close(fig)

    def _errors(
        self, random_style: bool, mask_random: Masker, mask_island: Masker
    ) -> tuple[list[float], list[float]]:
        errors_t_random_e_random, errors_t_random_e_island = [], []
        for __, learner in self.models(
            include_random_masks=random_style, include_island_masks=not random_style
        ):
            errors_t_random_e_random.append(
                learner.eval(
                    DatasetPoissonMaskedSolution(
                        self._grid_x1, self._grid_x2, self._ds_eval_raw, mask_random
                    ).make()
                )
            )
            errors_t_random_e_island.append(
                learner.eval(
                    DatasetPoissonMaskedSolution(
                        self._grid_x1, self._grid_x2, self._ds_eval_raw, mask_island
                    ).make()
                )
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
            style = {"linestyle": "dashed", "marker": "x"}
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
            ax.set_xlabel("masking proportion")
            ax.set_ylabel("error [$L^2$]")
            ax.set_title(
                "error VS masking \n"
                f"{learner.as_name()} on "
                f'["{self._dataset_name}" with "{dataset_train.as_name()}"]'
            )
            ax.legend()

            saveload.save(fig, location, overwrite=True)
            plt.close(fig)

    def _errors_eval(
        self, learner: LearnerPoissonFourier, random_style: bool = True
    ) -> typing.Sequence[float]:
        errors = []
        for ds in self.datasets_masked(random_style=random_style):
            errors.append(learner.eval(ds.make()))
        return errors


class PipelineFNOMaskedSolution(PipelineMaskedSolution):
    def __init__(self):
        super().__init__(network_name="fno")

    def _model_trained(
        self, dataset: DatasetMasked, batch_size: int = 20
    ) -> LearnerPoissonFNOMaskedSolution:
        location = self._name_model(dataset.as_name())
        if self._saveload.exists(location):
            learner = LearnerPoissonFNOMaskedSolution(
                self._grid_x1,
                self._grid_x2,
                self._saveload.load(location),
            )
        else:
            network = fno_2d.FNO2d(n_channels_lhs=4, n_channels_rhs=1)
            learner = LearnerPoissonFNOMaskedSolution(
                self._grid_x1,
                self._grid_x2,
                network,
            )
            learner.train(dataset.make(), n_epochs=1001, batch_size=batch_size)
            self._saveload.save(network, location)
        return learner


class PipelineCNOMaskedSolution(PipelineMaskedSolution):
    def __init__(self):
        super().__init__(network_name="cno")

    def _model_trained(
        self, dataset: DatasetMasked, batch_size: int = 20
    ) -> LearnerPoissonCNOMaskedSolution:
        location = self._name_model(dataset.as_name())
        if self._saveload.exists(location):
            learner = LearnerPoissonCNOMaskedSolution(
                self._grid_x1,
                self._grid_x2,
                self._saveload.load(location),
            )
        else:
            network = cno.CNO2d(in_channel=4, out_channel=1)
            learner = LearnerPoissonCNOMaskedSolution(
                self._grid_x1,
                self._grid_x2,
                network,
            )
            learner.train(dataset.make(), n_epochs=1001, batch_size=batch_size)
            self._saveload.save(network, location)
        return learner


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

    @abc.abstractmethod
    def _model_trained(
        self, dataset: DatasetMasked
    ) -> LearnerPoissonFNOMaskedSolutionSource:
        location = self._name_model(dataset.as_name())
        if self._saveload.exists(location):
            learner = LearnerPoissonFNOMaskedSolutionSource(
                self._grid_x1,
                self._grid_x2,
                self._saveload.load(location),
            )
        else:
            network = fno_2d.FNO2d(n_channels_lhs=4, n_channels_rhs=2)
            learner = LearnerPoissonFNOMaskedSolutionSource(
                self._grid_x1,
                self._grid_x2,
                network,
            )
            learner.train(dataset.make(), n_epochs=1001)
            self._saveload.save(network, location)
        return learner

    # def print_result_eval(self) -> None:
    #     for dataset_train, learner in self.models():
    #         print(f"eval> model trained on {dataset_train.as_name()}")
    #         for dataset_eval in self.datasets_masked(from_eval=False):
    #             print(f"eval> dataset {dataset_eval.as_name()}")
    #             learner.eval(dataset_eval.make(), print_result=True)
    #             print()
    #         print()

    def print_result_eval(self) -> None:
        for dataset_train, model in self.models():
            print(f"eval> {dataset_train.as_name()}")
            # for perc in self._percs:
            #     self._eval_one(
            #         model,
            #         DatasetPoissonMaskedSolutionSource(
            #             self._grid_x1,
            #             self._grid_x2,
            #             self._ds_eval_raw,
            #             mask_solution=MaskerRandom(perc),
            #             mask_source=MaskerRandom(perc),
            #         ),
            #     )
            for perc in self._percs:
                learner.eval(
                    DatasetPoissonMaskedSolutionSource(
                        self._grid_x1,
                        self._grid_x2,
                        self._ds_eval_raw,
                        mask_solution=MaskerIsland(1 - perc),
                        mask_source=MaskerIsland(1 - perc),
                    ).make(),
                    print_result=True,
                )

    def _eval_one(
        self,
        learner: LearnerPoissonFourier,
        dataset_eval: DatasetMasked,
    ) -> None:
        learner.eval(dataset_eval.make())

    def plot_mask(self) -> None:
        for dataset_train, model in self.models():
            for perc in self._percs:
                self._plot_compare(
                    learner,
                    dataset_train,
                    DatasetPoissonMaskedSolutionSource(
                        self._grid_x1,
                        self._grid_x2,
                        self._ds_eval_raw,
                        mask_solution=MaskerRandom(perc_to_mask=perc),
                        mask_source=MaskerRandom(perc_to_mask=perc),
                    ),
                )

    def _plot_mask_to_error(
        self,
        percs_to_mask: np.ndarray,
        errors: list[float],
        name_problem: str,
        name_model: str,
        name_mask: str,
    ) -> None:
        fig, ax = plt.subplots()
        style = {"linestyle": "dashed", "marker": "x"}
        ax.plot(percs_to_mask, errors, **style)
        ax.set_xlabel(f"masking proportion [{name_mask}-style]")
        ax.set_ylabel("error [L2]")
        ax.set_title(f"error VS masking [{name_model}]")

        saveload = SaveloadImage(self._saveload_base)
        location = f"mask_to_error--{name_problem}--{name_model}"
        if name_mask:
            location = f"{location}--{name_mask}"
        saveload.save(fig, saveload.rebase_location(location), overwrite=True)


class PipelineFNOMaskedSolutionSource(PipelineMaskedSolutionSource):
    def __init__(self):
        super().__init__(network_name="fno")

    def _model_trained(
        self, dataset: DatasetMasked
    ) -> LearnerPoissonFNOMaskedSolutionSource:
        location = self._name_model(dataset.as_name())
        if self._saveload.exists(location):
            learner = LearnerPoissonFNOMaskedSolutionSource(
                self._grid_x1,
                self._grid_x2,
                self._saveload.load(location),
            )
        else:
            network = fno_2d.FNO2d(n_channels_lhs=4, n_channels_rhs=2)
            learner = LearnerPoissonFNOMaskedSolutionSource(
                self._grid_x1,
                self._grid_x2,
                network,
            )
            learner.train(dataset.make(), n_epochs=1001)
            self._saveload.save(network, location)
        return learner


class PipelineCNOMaskedSolutionSource(PipelineMaskedSolutionSource):
    def __init__(self):
        super().__init__(network_name="cno")

    def _model_trained(
        self, dataset: DatasetMasked
    ) -> LearnerPoissonCNOMaskedSolutionSource:
        location = self._name_model(dataset.as_name())
        if self._saveload.exists(location):
            learner = LearnerPoissonCNOMaskedSolutionSource(
                self._grid_x1,
                self._grid_x2,
                self._saveload.load(location),
            )
        else:
            network = cno.CNO2d(in_channel=4, out_channel=2)
            learner = LearnerPoissonCNOMaskedSolutionSource(
                self._grid_x1,
                self._grid_x2,
                network,
            )
            learner.train(dataset.make(), n_epochs=1001)
            self._saveload.save(network, location)
        return learner


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )

    torch.manual_seed(42)

    fnos_m = PipelineFNOMaskedSolution()
    fnos_m.train()
    fnos_m.print_result_eval()

    cnos_m = PipelineCNOMaskedSolution()
    cnos_m.train()
    cnos_m.print_result_eval()

    fnos_mm = PipelineFNOMaskedSolutionSource()
    fnos_mm.train()
    fnos_mm.print_result_eval()

    cnos_mm = PipelineCNOMaskedSolutionSource()
    cnos_mm.train()
    cnos_mm.print_result_eval()
