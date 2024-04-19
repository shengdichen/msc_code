import abc
import collections
import logging
import pathlib
import typing

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.deepl import cno, fno_2d, network
from src.definition import DEFINITION, T_DATASET, T_NETWORK
from src.numerics import distance, grid, multidiff
from src.pde.poisson.dataset import SolverPoisson
from src.util import plot
from src.util.saveload import SaveloadImage, SaveloadTorch

logger = logging.getLogger(__name__)


class LearnerPoissonFourier:
    def __init__(self, grids: grid.Grids, network_fno: T_NETWORK):
        self._device = DEFINITION.device_preferred

        self._grids = grids
        self._network = network_fno.to(self._device)

    @abc.abstractmethod
    def as_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def train(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
        batch_size: int = 2,
        n_epochs: int = 2001,
        freq_eval: int = 100,
        datasets_eval: typing.Optional[
            typing.Sequence[torch.utils.data.dataset.TensorDataset]
        ] = None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def eval(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
        print_result: bool = False,
    ) -> float:
        raise NotImplementedError

    def plot_comparison_2d(
        self, dataset: torch.utils.data.dataset.TensorDataset
    ) -> mpl.figure.Figure:
        u_theirs, u_ours = self._plotdata_u(dataset)

        fig, (ax_1, ax_2) = plt.subplots(
            1, 2, width_ratios=(1, 1), figsize=(10, 5), dpi=200
        )
        putil = plot.PlotUtil(self._grids)

        # TODO: how do we use colorbar?
        # fig.colorbar(label="u")
        putil.plot_2d(ax_1, u_theirs)
        ax_1.set_title("$u$ (theirs)")

        putil.plot_2d(ax_2, u_ours)
        ax_2.set_title("$u$ (ours)")
        return fig

    def plot_comparison_3d(
        self, dataset: torch.utils.data.dataset.TensorDataset
    ) -> mpl.figure.Figure:
        u_theirs, u_ours = self._plotdata_u(dataset)

        fig, (ax_1, ax_2) = plt.subplots(
            1,
            2,
            width_ratios=(1, 1),
            figsize=(10, 5),
            dpi=200,
            subplot_kw={"projection": "3d"},
        )
        putil = plot.PlotUtil(self._grids)

        putil.plot_3d(ax_1, u_theirs)
        ax_1.set_title("$u$ (theirs)")

        putil.plot_3d(ax_2, u_ours)
        ax_2.set_title("$u$ (ours)")
        return fig

    def _plotdata_u(
        self, dataset: torch.utils.data.dataset.TensorDataset
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lhss, rhss = self._one_lhss_rhss(dataset)
        u_theirs = self._extract_u(rhss)

        with torch.no_grad():
            lhss = lhss.to(device=self._device, dtype=torch.float)
            self._network.eval()
            u_ours = self._extract_u(self._network(lhss).detach().to("cpu"))

        return u_theirs, u_ours

    @abc.abstractmethod
    def _extract_u(self, rhss: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _one_lhss_rhss(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
        index: int = 0,
        flatten_first_dimension: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lhss, rhss = list(dataset)[index]
        if not flatten_first_dimension:
            return lhss.unsqueeze(0), rhss.unsqueeze(0)
        return lhss, rhss

    def iterate_dataset(
        self, dataset: torch.utils.data.dataset.TensorDataset, batch_size: int = 1
    ) -> collections.abc.Generator[
        tuple[np.ndarray, torch.Tensor, torch.Tensor], None, None
    ]:
        for lhss, rhss_theirs in torch.utils.data.DataLoader(
            dataset, batch_size=batch_size
        ):
            lhss, rhss_theirs = (
                lhss.to(device=self._device, dtype=torch.float),
                rhss_theirs.to(device=self._device, dtype=torch.float),
            )
            yield lhss, rhss_theirs, self._network(lhss)

    @abc.abstractmethod
    def errors(
        self, dataset: torch.utils.data.dataset.TensorDataset
    ) -> typing.Sequence[float]:
        raise NotImplementedError


class LearnerPoissonFNOMaskedSolution(LearnerPoissonFourier):
    def __init__(self, grids: grid.Grids, network_fno: fno_2d.FNO2d):
        super().__init__(grids, network_fno)

    def as_name(self) -> str:
        return "fno-2d"

    def train(
        self,
        dataset: T_DATASET,
        batch_size: int = 30,
        n_epochs: int = 1001,
        freq_eval: int = 100,
        datasets_eval: typing.Optional[typing.Sequence[T_DATASET]] = None,
    ) -> None:
        optimizer = torch.optim.Adam(self._network.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,
            gamma=0.5,  # more conservative decay than default
        )

        if (n_epochs % freq_eval) == 0:
            n_epochs += 1  # one extra final eval report

        for epoch in tqdm(range(n_epochs)):
            self._network.train()
            mse_abs_all, mse_rel_all = [], []
            for __, rhss_theirs, rhss_ours in self.iterate_dataset(dataset, batch_size):
                dst = distance.Distance(rhss_theirs, rhss_ours)
                mse_abs = dst.mse()
                mse_abs.backward()
                optimizer.step()
                optimizer.zero_grad()

                mse_abs_all.append(mse_abs.item())
                mse_rel_all.append(dst.mse_relative().item())

            scheduler.step()
            if epoch % freq_eval == 0:
                print(
                    "train> (mse, mse%): "
                    f"{np.average(mse_abs_all):.4}, {np.average(mse_rel_all):.4%}"
                )
                if datasets_eval:
                    for dataset_eval in datasets_eval:
                        self.eval(dataset_eval, print_result=True)
                print()

    def eval(
        self,
        dataset: T_DATASET,
        print_result: bool = False,
    ) -> float:
        mse_abs_all, mse_rel_all = [], []
        self._network.eval()
        with torch.no_grad():
            for __, rhss_theirs, rhss_ours in self.iterate_dataset(
                dataset, batch_size=30
            ):
                dst = distance.Distance(rhss_ours, rhss_theirs)
                mse_abs_all.append(dst.mse().item())
                mse_rel_all.append(dst.mse_relative().item())

        mse_abs_avg, mse_rel_avg = np.average(mse_abs_all), np.average(mse_rel_all)
        if print_result:
            print(f"eval> (mse, mse%): {mse_abs_avg:.4}, {mse_rel_avg:.4%}")

        return mse_rel_avg.item()

    def errors(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
    ) -> typing.Sequence[float]:
        mse_rel_all = []
        with torch.no_grad():
            self._network.eval()
            for __, rhss_theirs, rhss_ours in self.iterate_dataset(
                dataset, batch_size=30
            ):
                mse_rel_all.append(
                    distance.Distance(rhss_ours, rhss_theirs).mse_relative().item()
                )

        self._network.train()
        return (np.average(mse_rel_all),)

    def _extract_u(self, rhss: torch.Tensor) -> torch.Tensor:
        return rhss[0, 0, :, :]


class LearnerPoissonFNOMaskedSolutionSource(LearnerPoissonFourier):
    def __init__(self, grids: grid.Grids, network_fno: fno_2d.FNO2d):
        super().__init__(grids, network_fno)

    def as_name(self) -> str:
        return "fno-2d"

    def train(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
        batch_size: int = 2,
        n_epochs: int = 2001,
        freq_eval: int = 100,
        datasets_eval: typing.Optional[
            typing.Sequence[torch.utils.data.dataset.TensorDataset]
        ] = None,
    ) -> None:
        optimizer = torch.optim.Adam(self._network.parameters(), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        for epoch in range(n_epochs):
            mse_solution, mse_source, mse_all = [], [], []
            for __, rhss_theirs, rhss_ours in self.iterate_dataset(dataset, batch_size):
                dst_solutions, dst_sources = self._calc_distances(
                    rhss_ours, rhss_theirs
                )
                mse_abs = self._calc_loss(dst_solutions, dst_sources)
                mse_abs.backward()
                optimizer.step()
                optimizer.zero_grad()

                mse_solution.append(dst_solutions.mse_relative().item())
                mse_source.append(dst_sources.mse_relative().item())
                mse_all.append(mse_abs.item())

            scheduler.step()
            if epoch % freq_eval == 0:
                print(
                    "train> (solution%, source%; all): "
                    f"{np.average(mse_solution)}, {np.average(mse_source)}; "
                    f"{np.average(mse_all)}"
                )
                if datasets_eval:
                    for dataset_eval in datasets_eval:
                        self.eval(dataset_eval, print_result=True)
                print()

    def eval(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
        print_result: bool = False,
    ) -> float:
        mse_solution, mse_source, mse_all = [], [], []
        with torch.no_grad():
            self._network.eval()
            for __, rhss_theirs, rhss_ours in self.iterate_dataset(
                dataset, batch_size=1
            ):
                dst_solutions, dst_sources = self._calc_distances(
                    rhss_ours, rhss_theirs
                )
                mse_solution.append(dst_solutions.mse_relative().item())
                mse_source.append(dst_sources.mse_relative().item())
                mse_all.append(self._calc_loss(dst_sources, dst_sources).item())

        self._network.train()
        mse_all_avg = np.average(mse_all)
        if print_result:
            print(
                "eval> (solution%, source%; all): "
                f"{np.average(mse_solution)}, {np.average(mse_source)}; "
                f"{mse_all_avg}"
            )
        return mse_all_avg.item()

    def errors(
        self, dataset: torch.utils.data.dataset.TensorDataset
    ) -> typing.Sequence[float]:
        mse_solution, mse_source = [], []
        with torch.no_grad():
            self._network.eval()
            for __, rhss_theirs, rhss_ours in self.iterate_dataset(
                dataset, batch_size=1
            ):
                dst_solutions, dst_sources = self._calc_distances(
                    rhss_ours, rhss_theirs
                )
                mse_solution.append(dst_solutions.mse_relative().item())
                mse_source.append(dst_sources.mse_relative().item())

        self._network.train()
        return np.average(mse_solution), np.average(mse_source)

    def _calc_distances(
        self, rhss_ours: torch.Tensor, rhss_theirs: torch.Tensor
    ) -> tuple[distance.Distance, distance.Distance]:
        solutions_ours, sources_ours = rhss_ours[:, :, :, 0], rhss_ours[:, :, :, 1]
        solutions_theirs, sources_theirs = (
            rhss_theirs[:, :, :, 0],
            rhss_theirs[:, :, :, 1],
        )
        return (
            distance.Distance(solutions_ours, solutions_theirs),
            distance.Distance(sources_theirs, sources_ours),
        )

    def _calc_loss(
        self, dst_solutions: distance.Distance, dst_sources: distance.Distance
    ) -> torch.Tensor:
        return dst_solutions.mse() + 0.7 * dst_sources.mse()

    def _extract_u(self, rhss: torch.Tensor) -> torch.Tensor:
        return rhss[0, :, :, 0]


class DatasetReorderCNO:
    def __init__(self, dataset: torch.utils.data.dataset.TensorDataset):
        self._dataset = dataset

    def reorder(self) -> torch.utils.data.dataset.TensorDataset:
        lhss, rhss = [], []
        for lhs, rhs in self._dataset:
            lhss.append(lhs.permute(2, 0, 1))
            rhss.append(rhs.permute(2, 0, 1))
        return torch.utils.data.TensorDataset(torch.stack(lhss), torch.stack(rhss))


class LearnerPoissonCNOMaskedSolution(LearnerPoissonFNOMaskedSolution):
    def __init__(self, grid_x1: grid.Grid, grid_x2: grid.Grid, network_cno: cno.CNO2d):
        super().__init__(grid_x1, grid_x2, network_cno)

    def as_name(self) -> str:
        return "cno-2d"

    def train(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
        batch_size: int = 2,
        n_epochs: int = 2001,
        freq_eval: int = 100,
        dataset_eval: typing.Optional[torch.utils.data.dataset.TensorDataset] = None,
    ) -> None:
        return super().train(
            DatasetReorderCNO(dataset).reorder(),
            batch_size,
            n_epochs,
            freq_eval,
            dataset_eval,
        )

    def eval(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
        print_result: bool = False,
    ) -> float:
        return super().eval(DatasetReorderCNO(dataset).reorder(), print_result)

    def errors(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
    ) -> typing.Sequence[float]:
        return super().errors(DatasetReorderCNO(dataset).reorder())

    def _extract_u(self, rhss: torch.Tensor) -> torch.Tensor:
        return rhss[0, 0, :, :]


class LearnerPoissonCNOMaskedSolutionSource(LearnerPoissonFNOMaskedSolutionSource):
    def __init__(self, grid_x1: grid.Grid, grid_x2: grid.Grid, network_cno: cno.CNO2d):
        super().__init__(grid_x1, grid_x2, network_cno)

    def as_name(self) -> str:
        return "cno-2d"

    def train(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
        batch_size: int = 2,
        n_epochs: int = 2001,
        freq_eval: int = 100,
        dataset_eval: typing.Optional[torch.utils.data.dataset.TensorDataset] = None,
    ) -> None:
        return super().train(
            DatasetReorderCNO(dataset).reorder(),
            batch_size,
            n_epochs,
            freq_eval,
            dataset_eval,
        )

    def eval(
        self,
        dataset: torch.utils.data.dataset.TensorDataset,
        print_result: bool = False,
    ) -> float:
        return super().eval(DatasetReorderCNO(dataset).reorder(), print_result)

    def _extract_u(self, rhss: torch.Tensor) -> torch.Tensor:
        return rhss[0, 0, :, :]

    def errors(
        self, dataset: torch.utils.data.dataset.TensorDataset
    ) -> typing.Sequence[float]:
        return super().errors(DatasetReorderCNO(dataset).reorder())


class LearnerPoissonFC:
    def __init__(self, n_pts_mask: int = 30):
        self._device = DEFINITION.device_preferred
        self._name_variant = f"fc-{n_pts_mask}"

        grid_x1 = grid.Grid(n_pts=50, stepsize=0.1, start=0.0)
        grid_x2 = grid.Grid(n_pts=50, stepsize=0.1, start=0.0)
        self._grids_full = grid.Grids([grid_x1, grid_x2])

        self._saveload = SaveloadTorch("poisson")
        solver = SolverPoisson(
            grid_x1, grid_x2, source=self._grids_full.constants_like(-200)
        )
        self._solver = solver.as_interpolator(
            saveload=self._saveload,
            name="dataset-fc",
            boundary_mean=-20,
            boundary_sigma=1,
        )
        self._lhss_eval, self._rhss_exact_eval = self._make_lhss_rhss_train(
            self._grids_full, n_pts=5000
        )

        grids_mask = grid.Grids(
            [
                grid.Grid(n_pts=n_pts_mask, stepsize=0.1, start=0.5),
                grid.Grid(n_pts=n_pts_mask, stepsize=0.1, start=0.5),
            ]
        )
        self._lhss_train, self._rhss_exact_train = self._make_lhss_rhss_train(
            grids_mask, n_pts=4000
        )

        self._network = network.Network(dim_x=2, with_time=False).to(self._device)
        self._eval_network = self._make_eval_network(use_multidiff=False)

    def _make_lhss_rhss_train(
        self, grids: grid.Grids, n_pts: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lhss = grids.samples_sobol(n_pts)
        rhss = torch.from_numpy(self._solver.ev(lhss[:, 0], lhss[:, 1])).view(-1, 1)
        return lhss.to(self._device), rhss.to(self._device)

    def _make_eval_network(
        self, use_multidiff: bool = True
    ) -> typing.Callable[[torch.Tensor], torch.Tensor]:
        def f(lhss: torch.Tensor) -> torch.Tensor:
            if use_multidiff:
                mdn = multidiff.MultidiffNetwork(self._network, lhss, ["x1", "x2"])
                return mdn.diff("x1", 2) + mdn.diff("x2", 2)
            return self._network(lhss)

        return f

    def train(self, n_epochs: int = 50001) -> None:
        optimiser = torch.optim.Adam(self._network.parameters())
        for epoch in range(n_epochs):
            optimiser.zero_grad()
            loss = distance.Distance(
                self._eval_network(self._lhss_train), self._rhss_exact_train
            ).mse()
            loss.backward()
            optimiser.step()

            if epoch % 100 == 0:
                logger.info(f"epoch {epoch:04}> " f"loss [train]: {loss.item():.4} ")
                self.evaluate_model()

        self._saveload.save(
            self._network,
            self._saveload.rebase_location(f"network-{self._name_variant}"),
        )

    def evaluate_model(self) -> None:
        dist = distance.Distance(
            self._eval_network(self._lhss_eval), self._rhss_exact_eval
        )
        logger.info(f"eval> (mse, mse%): {dist.mse()}, {dist.mse_percentage()}")

    def load(self) -> None:
        location = self._saveload.rebase_location(f"network-{self._name_variant}")
        self._network = self._saveload.load(location)

    def plot(self) -> None:
        res = self._grids_full.zeroes_like_numpy()

        for (
            (idx_x1, val_x1),
            (idx_x2, val_x2),
        ) in self._grids_full.steps_with_index():
            lhs = torch.tensor([val_x1, val_x2]).view(1, 2).to(self._device)
            res[idx_x1, idx_x2] = self._eval_network(lhs)

        plotter = plot.PlotFrame(
            self._grids_full,
            res,
            f"poisson-{self._name_variant}",
            SaveloadImage(self._saveload.base),
        )
        plotter.plot_2d()
        plotter.plot_3d()
