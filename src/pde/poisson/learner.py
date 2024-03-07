import abc
import logging
import typing

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.deepl import cno, fno_2d, network
from src.definition import DEFINITION
from src.numerics import distance, grid, multidiff
from src.pde.poisson.dataset import SolverPoisson
from src.util import plot
from src.util.saveload import SaveloadImage, SaveloadTorch

logger = logging.getLogger(__name__)


class LearnerPoissonFNO:
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        network_fno: torch.nn.Module,
        dataset_eval: torch.utils.data.dataset.TensorDataset,
        dataset_train: torch.utils.data.dataset.TensorDataset,
        saveload: SaveloadTorch,
        name_learner: str,
    ):
        self._device = DEFINITION.device_preferred

        self._grid_x1, self._grid_x2 = grid_x1, grid_x2
        self._grids = grid.Grids([self._grid_x1, self._grid_x2])
        self._network = network_fno.to(self._device)
        self._dataset_eval, self._dataset_train = dataset_eval, dataset_train

        self._saveload, self._name_learner = saveload, name_learner
        self._location = self._saveload.rebase_location(name_learner)

    def train(self, n_epochs: int = 2001, freq_eval: int = 100) -> None:
        optimizer = torch.optim.Adam(self._network.parameters(), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        for epoch in range(n_epochs):
            loss_all = []
            for lhss_batch, rhss_batch in torch.utils.data.DataLoader(
                self._dataset_train, batch_size=2
            ):
                lhss_batch, rhss_batch = (
                    lhss_batch.to(device=self._device, dtype=torch.float),
                    rhss_batch.to(device=self._device, dtype=torch.float),
                )
                optimizer.zero_grad()
                rhss_ours = self._network(lhss_batch)  # 10, 1001, 1
                loss_batch = distance.Distance(rhss_ours, rhss_batch).mse()
                loss_batch.backward()
                optimizer.step()

                loss_all.append(loss_batch.item())

            scheduler.step()
            if epoch % freq_eval == 0:
                print(f"train> mse: {np.average(loss_all)}")
                self.eval()

    def load_network_trained(
        self, n_epochs: int = 2001, freq_eval: int = 100, save_as_suffix: str = "model"
    ) -> torch.nn.Module:
        def make() -> torch.nn.Module:
            self.train(n_epochs=n_epochs, freq_eval=freq_eval)
            return self._network

        location = (
            self._saveload.rebase_location(f"{self._name_learner}--{save_as_suffix}")
            if save_as_suffix
            else self._location
        )
        self._network = self._saveload.load_or_make(location, make)
        return self._network

    def eval(self, print_result: bool = True) -> float:
        mse_abs_all, mse_rel_all = [], []
        with torch.no_grad():
            self._network.eval()
            for lhss_batch, rhss_batch in torch.utils.data.DataLoader(
                self._dataset_eval
            ):
                lhss_batch, rhss_batch = (
                    lhss_batch.to(device=self._device, dtype=torch.float),
                    rhss_batch.to(device=self._device, dtype=torch.float),
                )
                rhss_ours = self._network(lhss_batch)
                dst = distance.Distance(rhss_ours, rhss_batch)
                mse_abs_all.append(dst.mse().item())
                mse_rel_all.append(dst.mse_relative().item())
        mse_abs_avg, mse_rel_avg = np.average(mse_abs_all), np.average(mse_rel_all)
        if print_result:
            print(f"eval> (mse, mse%): {mse_abs_avg}, {mse_rel_avg}")

        return mse_rel_avg.item()

    def plot(self) -> mpl.figure.Figure:
        return self.plot_comparison_2d()

    def plot_comparison_2d(self) -> mpl.figure.Figure:
        u_theirs, u_ours = self._plotdata_u()

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

    def plot_comparison_3d(self) -> mpl.figure.Figure:
        u_theirs, u_ours = self._plotdata_u()

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

    @abc.abstractmethod
    def _plotdata_u(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _separate_lhss_rhss(
        self, dataset: torch.utils.data.dataset.TensorDataset
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lhss, rhss = [], []
        for lhs, rhs in dataset:
            lhss.append(lhs)
            rhss.append(rhs)
        return torch.stack(lhss), torch.stack(rhss)

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


class LearnerPoissonFNO2d(LearnerPoissonFNO):
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        dataset_eval: torch.utils.data.dataset.TensorDataset,
        dataset_train: torch.utils.data.dataset.TensorDataset,
        saveload: SaveloadTorch,
        name_learner: str,
    ):
        super().__init__(
            grid_x1,
            grid_x2,
            fno_2d.FNO2d(n_channels_lhs=4),
            dataset_eval=dataset_eval,
            dataset_train=dataset_train,
            saveload=saveload,
            name_learner=f"network_fno_2d--{name_learner}",
        )

    def _plotdata_u(self) -> tuple[torch.Tensor, torch.Tensor]:
        lhss, rhss = self._one_lhss_rhss(self._dataset_eval)
        u_theirs = rhss[0, :, :, 0]

        with torch.no_grad():
            lhss = lhss.to(device=self._device, dtype=torch.float)
            self._network.eval()
            u_ours = self._network(lhss).detach().to("cpu")[0, :, :, 0]

        return u_theirs, u_ours


class DatasetReorderCNO:
    def __init__(self, dataset: torch.utils.data.dataset.TensorDataset):
        self._dataset = dataset

    def reorder(self) -> torch.utils.data.dataset.TensorDataset:
        lhss, rhss = [], []
        for lhs, rhs in self._dataset:
            lhss.append(lhs.permute(2, 0, 1))
            rhss.append(rhs.permute(2, 0, 1))
        return torch.utils.data.TensorDataset(torch.stack(lhss), torch.stack(rhss))


class LearnerPoissonCNO2d(LearnerPoissonFNO):
    def __init__(
        self,
        grid_x1: grid.Grid,
        grid_x2: grid.Grid,
        dataset_eval: torch.utils.data.dataset.TensorDataset,
        dataset_train: torch.utils.data.dataset.TensorDataset,
        saveload: SaveloadTorch,
        name_learner: str,
    ):
        super().__init__(
            grid_x1,
            grid_x2,
            cno.CNO2d(in_channel=4, out_channel=1),
            dataset_eval=DatasetReorderCNO(dataset_eval).reorder(),
            dataset_train=DatasetReorderCNO(dataset_train).reorder(),
            saveload=saveload,
            name_learner=f"network_cno_2d--{name_learner}",
        )

    def _plotdata_u(self) -> tuple[torch.Tensor, torch.Tensor]:
        lhss, rhss = self._one_lhss_rhss(self._dataset_eval)
        u_theirs = rhss[0, 0, :, :]

        with torch.no_grad():
            lhss = lhss.to(device=self._device, dtype=torch.float)
            self._network.eval()
            u_ours = self._network(lhss).detach().to("cpu")[0, 0, :, :]

        return u_theirs, u_ours


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
