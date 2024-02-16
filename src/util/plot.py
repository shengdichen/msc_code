import torch
from matplotlib import pyplot as plt

from src.numerics import grid
from src.util.saveload import SaveloadImage


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

    def plot_2d(self) -> None:
        plt.figure(figsize=(8, 6))

        plt.contourf(self._coords_x1, self._coords_x2, self._sol, cmap="viridis")
        plt.colorbar(label="u(x, y)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

        title = f"{self._name}-2d"
        plt.title(title)
        self._saveload.save(plt, self._saveload.rebase_location(title))

    def plot_3d(self) -> None:
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
        self._saveload.save(plt, self._saveload.rebase_location(title))


class PlotMovie:
    pass
