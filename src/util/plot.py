import matplotlib as mpl
import torch
from matplotlib import pyplot as plt

from src.numerics import grid
from src.util.saveload import SaveloadImage


class PlotUtil:
    def __init__(
        self,
        grids: grid.Grids,
    ):
        self._grids = grids
        self._coords_x1, self._coords_x2 = self._grids.coords_as_mesh()

    def plot_2d(self, ax: mpl.axes.Axes, target: torch.Tensor) -> None:
        ax.contourf(self._coords_x1, self._coords_x2, target, cmap="viridis")
        self._set_label_xy(ax)
        ax.grid(True)

    def plot_3d(self, ax: mpl.axes.Axes, target: torch.Tensor) -> None:
        surface = ax.plot_surface(
            self._coords_x1,
            self._coords_x2,
            target,
            cmap="viridis",
            edgecolor="k",
        )

        self._set_label_xy(ax)
        ax.set_zlabel("$u$")

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


class PlotMovie:
    pass
