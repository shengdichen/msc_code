import abc
import typing

import torch

from src.numerics import grid


class DatasetPDE:
    def __init__(self, grids: grid.Grids):
        self._grids = grids

    @abc.abstractmethod
    def solve(self, n_instances: int) -> typing.Iterable[typing.Iterable[torch.Tensor]]:
        for __ in range(n_instances):
            yield self.solve_instance()

    @abc.abstractmethod
    def solve_instance(self) -> typing.Iterable[torch.Tensor]:
        raise NotImplementedError


class DatasetPDE2d(DatasetPDE):
    def __init__(self, grid_x1: grid.Grid, grid_x2: grid.Grid):
        self._grid_x1, self._grid_x2 = grid_x1, grid_x2

        super().__init__(grid.Grids([self._grid_x1, self._grid_x2]))

    @abc.abstractmethod
    def solve_instance(self) -> typing.Iterable[torch.Tensor]:
        raise NotImplementedError
