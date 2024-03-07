import logging
import typing

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.numerics import grid
from src.pde.poisson.dataset import (DatasetConstructed, DatasetConstructedSin,
                                     DatasetPoisson, DatasetSolver)
from src.pde.poisson.learner import (LearnerPoissonCNO2d, LearnerPoissonFNO,
                                     LearnerPoissonFNO2d)
from src.util.dataset import Masker, MaskerIsland, MaskerRandom
from src.util.saveload import SaveloadImage, SaveloadTorch

logger = logging.getLogger(__name__)


class DatasetSplits:
    def __init__(
        self,
        dataset_full: DatasetPoisson,
        n_instances_eval: int = 300,
        n_instances_train: int = 100,
    ):
        self._dataset_full = dataset_full
        self._n_instances = self._dataset_full.n_instances
        self._n_instances_eval, self._n_instances_train = (
            n_instances_eval,
            n_instances_train,
        )

    def split(
        self,
    ) -> tuple[
        torch.utils.data.dataset.TensorDataset, torch.utils.data.dataset.TensorDataset
    ]:
        indexes_eval, indexes_train = self._indexes_eval_train()

        return (
            self._dataset_full.dataset_raw_split(
                indexes=indexes_eval, save_as_suffix="eval"
            ),
            self._dataset_full.dataset_raw_split(
                indexes=indexes_train, save_as_suffix="train"
            ),
        )

    def _indexes_eval_train(self) -> tuple[np.ndarray, np.ndarray]:
        # NOTE:
        # generate indexes in one call with |replace| set to |False| to guarantee strict
        # separation of train and eval datasets
        indexes = np.random.default_rng(seed=42).choice(
            self._n_instances,
            self._n_instances_eval + self._n_instances_train,
            replace=False,
        )
        return (
            indexes[: self._n_instances_eval],
            indexes[-self._n_instances_train :],
        )


class Learners:
    def __init__(self, n_instances_eval: int = 300, n_instances_train=100):
        self._grid_x1 = grid.Grid(n_pts=64, stepsize=0.01, start=0.0)
        self._grid_x2 = grid.Grid(n_pts=64, stepsize=0.01, start=0.0)

        self._n_instances_eval, self._n_instances_train = (
            n_instances_eval,
            n_instances_train,
        )

        self._saveload_base = "poisson"
        self._saveload = SaveloadTorch(self._saveload_base)

    def dataset_standard(self) -> None:
        name_dataset = "dataset-fno-2d-standard"
        ds = DatasetSolver(
            self._grid_x1,
            self._grid_x2,
            saveload=self._saveload,
            name_dataset=name_dataset,
            source=grid.Grids([self._grid_x1, self._grid_x2]).constants_like(-200),
            boundary_mean=-20,
            boundary_sigma=1,
        )

        learner = LearnerPoissonFNO2d(
            self._grid_x1,
            self._grid_x2,
            dataset_eval=ds.dataset_masked(mask_solution=MaskerRandom(0.5)),
            dataset_train=ds.dataset_masked(mask_solution=MaskerRandom(0.5)),
            saveload=self._saveload,
            name_learner="standard",
        )
        learner.train()
        learner.plot()

    def dataset_custom_sin(
        self,
        ds_size: int = 1000,
        n_samples_per_instance: int = 3,
    ) -> None:
        name_problem = "custom_sin"
        dataset_full = DatasetConstructedSin(
            self._grid_x1,
            self._grid_x2,
            saveload=self._saveload,
            name_dataset=name_problem,
            n_instances=ds_size,
            n_samples_per_instance=n_samples_per_instance,
        )

        percs_to_mask = np.arange(start=0.1, stop=1.0, step=0.1)
        masks_random = [MaskerRandom(perc_to_mask=perc) for perc in percs_to_mask]
        masks_island = [MaskerIsland(perc_to_keep=1 - perc) for perc in percs_to_mask]

        self._plot_fnos(
            dataset_full,
            percs_to_mask,
            name_problem=name_problem,
            masks=masks_random,
            name_mask="random",
        )
        self._plot_fnos(
            dataset_full,
            percs_to_mask,
            name_problem=name_problem,
            masks=masks_island,
            name_mask="island",
        )
        self._plot_cnos(
            dataset_full,
            percs_to_mask,
            name_problem=name_problem,
            masks=masks_random,
            name_mask="random",
        )
        self._plot_cnos(
            dataset_full,
            percs_to_mask,
            name_problem=name_problem,
            masks=masks_island,
            name_mask="island",
        )

    def _plot_cnos(
        self,
        dataset_full: DatasetConstructed,
        percs_to_mask: np.ndarray,
        masks: typing.Sequence[Masker],
        name_mask: str,
        name_problem: str,
    ) -> None:
        errors = []
        ds_eval_raw, ds_train_raw = DatasetSplits(
            dataset_full,
            n_instances_eval=self._n_instances_eval,
            n_instances_train=self._n_instances_train,
        ).split()

        for perc, mask in zip(percs_to_mask, masks):
            ds_eval_masked = dataset_full.dataset_masked(
                from_dataset=ds_eval_raw,
                mask_solution=mask,
                save_as_suffix=f"eval_{self._n_instances_eval}",
            )
            ds_train_masked = dataset_full.dataset_masked(
                from_dataset=ds_train_raw,
                mask_solution=mask,
                save_as_suffix=f"train_{self._n_instances_train}",
            )
            learner = LearnerPoissonCNO2d(
                self._grid_x1,
                self._grid_x2,
                dataset_eval=ds_eval_masked,
                dataset_train=ds_train_masked,
                saveload=self._saveload,
                name_learner=name_problem,
            )
            detail_mask = f"{name_mask}_{perc:.2}"
            learner.load_network_trained(
                n_epochs=1001,
                save_as_suffix=detail_mask,
            )
            self._plot_comparison(
                learner,
                name_problem=name_problem,
                name_model="CNO",
                detail_mask=detail_mask,
            )
            errors.append(learner.eval(print_result=False))

        self._plot_mask_to_error(
            percs_to_mask,
            errors,
            name_problem=name_problem,
            name_model="CNO",
            name_mask=name_mask,
        )

    def _plot_fnos(
        self,
        dataset_full: DatasetConstructed,
        percs_to_mask: np.ndarray,
        masks: typing.Sequence[Masker],
        name_mask: str,
        name_problem: str,
    ) -> None:
        errors = []
        ds_eval_raw, ds_train_raw = DatasetSplits(
            dataset_full,
            n_instances_eval=self._n_instances_eval,
            n_instances_train=self._n_instances_train,
        ).split()

        for perc, mask in zip(percs_to_mask, masks):
            ds_eval_masked = dataset_full.dataset_masked(
                from_dataset=ds_eval_raw,
                mask_solution=mask,
                save_as_suffix=f"eval_{self._n_instances_eval}",
            )
            ds_train_masked = dataset_full.dataset_masked(
                from_dataset=ds_train_raw,
                mask_solution=mask,
                save_as_suffix=f"train_{self._n_instances_train}",
            )
            learner = LearnerPoissonFNO2d(
                self._grid_x1,
                self._grid_x2,
                dataset_eval=ds_eval_masked,
                dataset_train=ds_train_masked,
                saveload=self._saveload,
                name_learner=name_problem,
            )
            detail_mask = f"{name_mask}_{perc:.2}"
            learner.load_network_trained(
                n_epochs=1001,
                save_as_suffix=detail_mask,
            )
            self._plot_comparison(
                learner,
                name_problem=name_problem,
                name_model="FNO",
                detail_mask=detail_mask,
            )
            errors.append(learner.eval(print_result=False))

        self._plot_mask_to_error(
            percs_to_mask,
            errors,
            name_problem=name_problem,
            name_model="FNO",
            name_mask=name_mask,
        )

    def _plot_comparison(
        self,
        learner: LearnerPoissonFNO,
        name_problem: str,
        name_model: str,
        detail_mask: str,
    ) -> None:
        saveload = SaveloadImage(self._saveload_base)
        location = f"{name_problem}--{name_model}--{detail_mask}"
        saveload.save(
            learner.plot_comparison_2d(),
            saveload.rebase_location(f"{location}--2d"),
            overwrite=True,
        )
        saveload.save(
            learner.plot_comparison_3d(),
            saveload.rebase_location(f"{location}--3d"),
            overwrite=True,
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


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )

    torch.manual_seed(42)

    learners = Learners()
    learners.dataset_custom_sin()
