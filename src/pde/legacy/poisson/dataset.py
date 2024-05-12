import pathlib
import typing

import matplotlib as mpl
import torch
from matplotlib import pyplot as plt

from src.definition import T_DATASET
from src.numerics import grid
from src.pde.poisson.dataset import DatasetPoisson2d
from src.util import dataset as dataset_util
from src.util import plot


class DatasetPoissonMaskedSolution:
    N_CHANNELS_LHS = 8
    N_CHANNELS_RHS = 1

    def __init__(
        self,
        grids: grid.Grids,
        dataset_raw: DatasetPoisson2d,
        mask: dataset_util.Masker,
    ):
        self._grids = grids
        self._coords = self._grids.coords_as_mesh_torch()
        self._cos_coords = self._grids.cos_coords_as_mesh_torch()
        self._sin_coords = self._grids.sin_coords_as_mesh_torch()

        self._dataset_raw = dataset_raw
        self._mask = mask
        self._name = f"{self._dataset_raw.name_dataset}--sol_{self._mask.name}"

        self._normalizer: dataset_util.Normalizer
        self._dataset: T_DATASET

    @property
    def name(self) -> str:
        return self._name

    @property
    def dataset(self) -> T_DATASET:
        return self._dataset

    @classmethod
    def load_split(
        cls,
        grids: grid.Grids,
        dataset_raw: DatasetPoisson2d,
        masks_eval: typing.Iterable[dataset_util.Masker],
        masks_train: typing.Iterable[dataset_util.Masker],
        n_instances_eval: int,
        n_instances_train: int,
        base_dir: pathlib.Path = pathlib.Path("."),
    ) -> tuple[
        typing.Sequence["DatasetPoissonMaskedSolution"],
        typing.Sequence["DatasetPoissonMaskedSolution"],
    ]:
        eval_raw, train_raw = dataset_raw.load_split(
            n_instances_eval=n_instances_eval,
            n_instances_train=n_instances_train,
            base_dir=base_dir,
        )

        evals = []
        for mask in masks_eval:
            ds = cls(grids, dataset_raw, mask)
            ds.make(
                eval_raw,
                save_as=base_dir / f"{ds.name}--eval_{n_instances_eval}.pth",
            )
            evals.append(ds)

        trains = []
        for mask in masks_train:
            ds = cls(grids, dataset_raw, mask)
            ds.make(
                train_raw,
                save_as=base_dir / f"{ds.name}--train_{n_instances_train}.pth",
            )
            trains.append(ds)

        return evals, trains

    def make(
        self,
        dataset: T_DATASET,
        components_to_last: bool = False,
        save_as: typing.Optional[pathlib.Path] = None,
    ) -> T_DATASET:
        if save_as and save_as.exists():
            self._dataset = torch.load(save_as)
            return self._dataset

        lhss, rhss = [], []
        for solution, source in dataset:
            lhss.append(
                torch.stack(
                    [
                        *self._coords,
                        *self._cos_coords,
                        *self._sin_coords,
                        solution,
                        source,
                    ]
                )
            )
            rhss.append(solution.unsqueeze(0))
        dataset = torch.utils.data.TensorDataset(torch.stack(lhss), torch.stack(rhss))
        self._dataset = self._apply_mask(self._normalize(dataset), self._mask)
        if components_to_last:
            self._dataset = dataset_util.Reorderer().components_to_last(self._dataset)

        if save_as:
            torch.save(self._dataset, save_as)

        return self._dataset

    def remake(
        self,
        n_instances: int,
    ) -> typing.Callable[[], T_DATASET]:
        def f() -> T_DATASET:
            return self.make(self._dataset_raw.as_dataset(n_instances))

        return f

    def _normalize(self, dataset: T_DATASET) -> T_DATASET:
        self._normalizer = dataset_util.Normalizer.from_dataset(dataset)
        return self._normalizer.normalize_dataset(dataset)

    def _apply_mask(self, dataset: T_DATASET, mask: dataset_util.Masker) -> T_DATASET:
        lhss, rhss = dataset_util.DatasetPde.from_dataset(dataset).lhss_rhss
        for lhs in lhss:
            lhs[-2] = mask.mask(lhs[-2])
        return torch.utils.data.TensorDataset(lhss, rhss)

    def plot_instance(
        self, dataset: T_DATASET, n_instances: int = 1
    ) -> mpl.figure.Figure:
        fig, (axs_unmasked, axs_masked) = plt.subplots(
            2, n_instances, figsize=(10, 7.3), dpi=200, subplot_kw={"aspect": 1.0}
        )
        colormap = mpl.colormaps["viridis"]
        putil = plot.PlotUtil(self._grids)

        for i, (lhss, rhss) in enumerate(dataset):
            solution_unmasked, solution_masked = rhss[:, :, 0], lhss[:, :, 0]
            ax_unmasked, ax_masked = axs_unmasked[i], axs_masked[i]
            ax_unmasked.set_title(f"$u_{i+1}$")
            ax_masked.set_title(f"$u_{i+1}$ masked")
            putil.plot_2d(ax_unmasked, solution_unmasked, colormap=colormap)
            putil.plot_2d(ax_masked, solution_masked, colormap=colormap)
            if i == n_instances - 1:
                break
        return fig


class DatasetPoissonMaskedSolutionSource:
    N_CHANNELS_LHS = 8
    N_CHANNELS_RHS = 2

    def __init__(self, grids: grid.Grids):
        self._coords = grids.coords_as_mesh_torch()
        self._cos_coords = grids.cos_coords_as_mesh_torch()
        self._sin_coords = grids.sin_coords_as_mesh_torch()

        self._dataset = T_DATASET
        self._normalizer: dataset_util.Normalizer

    @staticmethod
    def as_name(
        dataset: DatasetPoisson2d,
        mask_solution: dataset_util.Masker,
        mask_source: dataset_util.Masker,
    ) -> str:
        return (
            f"{dataset.name_dataset}--"
            f"sol_{mask_solution.name()}--"
            f"source_{mask_source.name()}"
        )

    def make(
        self,
        dataset: T_DATASET,
        mask_solution: dataset_util.Masker,
        mask_source: dataset_util.Masker,
        components_to_last: bool = False,
        save_as: typing.Optional[pathlib.Path] = None,
    ) -> T_DATASET:
        if save_as and save_as.exists():
            return torch.load(save_as)

        lhss, rhss = [], []
        for solution, source in dataset:
            lhss.append(
                torch.stack(
                    [
                        *self._coords,
                        *self._cos_coords,
                        *self._sin_coords,
                        solution,
                        source,
                    ]
                )
            )
            rhss.append(torch.stack([solution, source]))
        dataset = torch.utils.data.TensorDataset(torch.stack(lhss), torch.stack(rhss))
        self._dataset = self._apply_mask(
            self._normalize(dataset), mask_solution, mask_source
        )
        if components_to_last:
            self._dataset = dataset_util.Reorderer().components_to_last(self._dataset)

        if save_as:
            torch.save(self._dataset, save_as)

        return self._dataset

    def _normalize(self, dataset: T_DATASET) -> T_DATASET:
        self._normalizer = dataset_util.Normalizer.from_dataset(dataset)
        return self._normalizer.normalize_dataset(dataset)

    def _apply_mask(
        self,
        dataset: T_DATASET,
        mask_solution: dataset_util.Masker,
        mask_source: dataset_util.Masker,
    ) -> T_DATASET:
        lhss, rhss = dataset_util.DatasetPde.from_dataset(dataset).lhss_rhss
        for lhs in lhss:
            lhs[-2] = mask_solution.mask(lhs[-2])
            lhs[-1] = mask_source.mask(lhs[-1])
        return torch.utils.data.TensorDataset(lhss, rhss)
