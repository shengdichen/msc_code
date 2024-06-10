import pathlib
import typing
from collections.abc import Sequence

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import animation
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from src.definition import DEFINITION
from src.numerics import grid
from src.util.saveload import SaveloadImage


class PlotUtil:
    def __init__(
        self,
        grids: grid.Grids,
    ):
        self._grids = grids
        self._coords_x1, self._coords_x2 = self._grids.coords_as_mesh()

    def plot_2d(
        self,
        ax: mpl.axes.Axes,
        target: torch.Tensor,
        colormap: typing.Optional[
            typing.Union[mpl.colors.ListedColormap, str]
        ] = "viridis",
    ) -> mpl.contour.QuadContourSet:
        self._set_label_xy(ax)
        ax.grid(True)
        colormap = colormap or "viridis"
        return ax.contourf(self._coords_x1, self._coords_x2, target, cmap=colormap)

    def plot_3d(
        self,
        ax: mpl.axes.Axes,
        target: torch.Tensor,
        show_label_xy: bool = True,
        label_z: str = "u",
    ) -> None:
        surface = ax.plot_surface(
            self._coords_x1,
            self._coords_x2,
            target,
            cmap="viridis",
            edgecolor="k",
        )

        if show_label_xy:
            self._set_label_xy(ax)
        if label_z:
            ax.set_zlabel(f"${label_z}$")

        return surface

    def _set_label_xy(self, ax: mpl.axes.Axes) -> None:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")


class PlotFrame:
    def __init__(
        self,
        grids: grid.Grids,
        sol: torch.Tensor,
        name: str,
        saveload: SaveloadImage,
    ):
        self._grids = grids
        self._coords_x1, self._coords_x2 = self._grids.coords_as_mesh()
        self._sol = sol

        self._saveload = saveload
        self._name = name

    def plot_2d(self, overwrite: bool = True) -> None:
        plt.figure(figsize=(8, 6))

        plt.contourf(self._coords_x1, self._coords_x2, self._sol, cmap="viridis")
        plt.colorbar(label="u(x, y)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

        title = f"{self._name}-2d"
        plt.title(title)
        self._saveload.save(
            plt, self._saveload.rebase_location(title), overwrite=overwrite
        )

    def plot_3d(self, overwrite: bool = True) -> None:
        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            self._coords_x1,
            self._coords_x2,
            self._sol,
            cmap="viridis",
            edgecolor="k",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("u(X, Y)")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        title = f"{self._name}-3d"
        ax.set_title(title)
        self._saveload.save(
            plt, self._saveload.rebase_location(title), overwrite=overwrite
        )


class PlotAnimation2D:
    def __init__(self, grid_time: grid.GridTime):
        self._grid_time = grid_time
        self._times = self._grid_time.step(with_start=True)

    def plot(
        self,
        matrices: typing.Sequence[np.ndarray],
        fig: mpl.figure.Figure,
        ax: mpl.axes.Axes,
        save_as: typing.Optional[pathlib.Path] = None,
    ) -> animation.FuncAnimation:
        image = ax.matshow(matrices[0], cmap="jet")

        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        fig.colorbar(image, cax=cax, orientation="vertical")

        def animate(frame: int):
            ax.set_title(
                f"time-step {self._grid_time.timestep_formatted(frame)}"
                "/"
                f"{self._grid_time.n_pts}"
                f" [$ t = {self._times[frame]:.4f} $] "
            )
            image.set_data(matrices[frame])
            return (image,)

        anm = animation.FuncAnimation(
            fig,
            animate,
            frames=tqdm(range(self._grid_time.n_pts)),
            interval=100,
            blit=True,
        )
        if save_as:
            anm.save(save_as)
        return anm


class PlotIllustration:
    def __init__(
        self,
        grids: grid.Grids,
        ticks_x1_2d: typing.Optional[Sequence[float]] = None,
        ticks_x2_2d: typing.Optional[Sequence[float]] = None,
        ticks_x1_3d: typing.Optional[Sequence[float]] = None,
        ticks_x2_3d: typing.Optional[Sequence[float]] = None,
    ):
        self._grids = grids
        self._coords_x1, self._coords_x2 = grids.coords_as_mesh()

        self._ticks_x1_2d, self._ticks_x2_2d = ticks_x1_2d, ticks_x2_2d
        self._ticks_x1_3d, self._ticks_x2_3d = ticks_x1_3d, ticks_x2_3d

        self._colormap, self._edgecolor = "viridis", "black"
        self._figure_height = {1: 4.5, 2: 8.5, 3: 13, 4: 16.5}

    def make_fig_ax(
        self, n_targets: int, title: typing.Optional[str] = None
    ) -> tuple[mpl.figure.Figure, tuple[list[mpl.axes.Axes], list[mpl.axes.Axes]]]:
        height = self._figure_height[n_targets]
        fig = plt.figure(figsize=(height, 9), dpi=200)
        if title:
            fig.suptitle(title)

        dim = 2, n_targets
        axs_2d = [fig.add_subplot(*dim, i + 1) for i in range(n_targets)]
        axs_3d = [
            fig.add_subplot(*dim, i + 1 + n_targets, projection="3d")
            for i in range(n_targets)
        ]

        return fig, (axs_2d, axs_3d)

    def plot_2d(
        self,
        ax: mpl.axes.Axes,
        target: np.ndarray,
        title: typing.Optional[str] = None,
        _min: typing.Optional[float] = None,
        _max: typing.Optional[float] = None,
    ) -> None:
        ax.contourf(
            self._coords_x1,
            self._coords_x2,
            target,
            vmin=_min,
            vmax=_max,
            cmap=self._colormap,
        )

        if title:
            ax.set_title(title)

        if self._ticks_x1_2d:
            ax.set_xticks(self._ticks_x1_2d)
        if self._ticks_x2_2d:
            ax.set_yticks(self._ticks_x2_2d)
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))

    def plot_3d(
        self,
        ax: mpl.axes.Axes,
        target: np.ndarray,
        _min: typing.Optional[float] = None,
        _max: typing.Optional[float] = None,
        ticks_z: typing.Optional[Sequence[float]] = None,
    ) -> None:
        ax.plot_surface(
            self._coords_x1,
            self._coords_x2,
            target,
            vmin=_min,
            vmax=_max,
            cmap=self._colormap,
            edgecolor=self._edgecolor,
        )

        ax.view_init(
            elev=33,  # tilted more towards us (default 30)
            azim=-63,  # rotated more counter-clockwise (default -60)
        )

        if self._ticks_x1_3d:
            ax.set_xticks(self._ticks_x1_3d)
        if self._ticks_x2_3d:
            ax.set_yticks(self._ticks_x2_3d)
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))

        ax.set_zlim(_min, _max)
        if ticks_z:
            ax.set_zticks(ticks_z)
        else:
            ax.zaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=4))
        ax.zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
        ax.zaxis.set_ticks_position("lower")  # ticks on left
        ax.tick_params(axis="z", labelrotation=17)

    def plot_targets(
        self,
        targets: typing.Sequence[np.ndarray],
        titles: typing.Sequence[str],
        _min: typing.Optional[float] = None,
        _max: typing.Optional[float] = None,
    ) -> tuple[mpl.figure.Figure, tuple[list[mpl.axes.Axes], list[mpl.axes.Axes]]]:
        fig, (axs_2d, axs_3d) = self.make_fig_ax(len(targets))

        for i, (target, title) in enumerate(zip(targets, titles)):
            self.plot_2d(axs_2d[i], target, title=title, _min=_min, _max=_max)
            self.plot_3d(axs_3d[i], target, _min=_min, _max=_max)

        return fig, (axs_2d, axs_3d)

    @staticmethod
    def plot_sample() -> None:
        DEFINITION.configure_font_matplotlib(font_latex=False)

        grids = grid.Grids(
            [
                grid.Grid.from_start_end(30, start=-np.pi, end=+np.pi),
                grid.Grid.from_start_end(30, start=-np.pi, end=+np.pi),
            ],
        )
        coords_x1, coords_x2 = grids.coords_as_mesh()
        targets = [
            np.sin(coords_x1) * np.sin(coords_x2),
            np.cos(coords_x1) * np.cos(coords_x2),
            np.sin(coords_x1) + np.sin(coords_x2),
            np.cos(coords_x1) - np.cos(coords_x2),
        ]
        titles = ["sinsin", "coscos", "sin + sin", "cos - cos"]

        plot = PlotIllustration(
            grids,
            ticks_x1_3d=[-3, -1, +1, +3],
            ticks_x2_3d=[-3, -1, +1, +3],
        )
        fig, (axs_2d, axs_3d) = plot.make_fig_ax(len(targets), title="Sample Plot")

        _min, _max = -1.0, +1.0
        ticks_z = list(np.arange(start=-0.75, stop=+0.75 + 0.1, step=0.5))
        plot.plot_2d(axs_2d[0], targets[0], title=titles[0], _min=_min, _max=_max)
        plot.plot_3d(axs_3d[0], targets[0], _min=_min, _max=_max, ticks_z=ticks_z)

        plot.plot_2d(axs_2d[1], targets[1], title=titles[1], _min=_min, _max=_max)
        plot.plot_3d(axs_3d[1], targets[1], _min=_min, _max=_max, ticks_z=ticks_z)

        _min, _max = -2.0, +2.0
        ticks_z = [-1.5, -0.5, +0.5, 1.5]
        plot.plot_2d(axs_2d[2], targets[2], title=titles[2], _min=_min, _max=_max)
        plot.plot_3d(axs_3d[2], targets[2], _min=_min, _max=_max, ticks_z=ticks_z)
        plot.plot_2d(axs_2d[3], targets[3], title=titles[3], _min=_min, _max=_max)
        plot.plot_3d(axs_3d[3], targets[3], _min=_min, _max=_max)

        PlotIllustration.finalize(fig, pathlib.Path("./sample_plot.png"))

    @staticmethod
    def finalize(fig: mpl.figure.Figure, save_as: pathlib.Path) -> None:
        fig.tight_layout()
        fig.savefig(save_as)
        plt.close(fig)


if __name__ == "__main__":
    PlotIllustration.plot_sample()
