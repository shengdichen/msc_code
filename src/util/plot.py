import pathlib
import typing

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import animation
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

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
