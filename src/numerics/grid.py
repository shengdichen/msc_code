import itertools
import math
from collections.abc import Iterable
from typing import Generator, Literal, Optional, Union

import numpy as np
import torch


class Grid:
    def __init__(self, n_pts: int, stepsize: float, start: float = 0.0):
        if n_pts < 1:
            raise ValueError("grid must have at least one grid-point")
        if stepsize <= 0.0:
            raise ValueError("stepsize must be positive")
        self._n_pts, self._stepsize = n_pts, stepsize

        self._start = start
        self._length = (n_pts - 1) * stepsize
        self._end = start + self._length
        self._pts = [self._start + i * self._stepsize for i in range(self._n_pts)]
        self._boundaries = [self._start, self._end]

    def __repr__(self):
        return (
            "grid: (start, end, stepsize, n_pts): "
            f"({self._start}, {self._end}, {self._stepsize}, {self._n_pts})"
        )

    @property
    def n_pts(self) -> int:
        return self._n_pts

    @property
    def stepsize(self) -> float:
        return self._stepsize

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    def step(self, with_start: bool = True, with_end: bool = True) -> list[float]:
        start = 0 if with_start else 1
        if not with_end:
            return self._pts[start:-1]
        return self._pts[start:]

    def step_with_index(
        self, with_start: bool = True, with_end: bool = True
    ) -> Iterable[tuple[int, float]]:
        start = 0 if with_start else 1
        return enumerate(self.step(with_start, with_end), start=start)

    def is_on_boundary(self, val: float) -> bool:
        for boundary in self._boundaries:
            if math.isclose(val, boundary):
                return True
        return False

    def is_in(self, value: float) -> bool:
        for pt in self._pts:
            if math.isclose(pt, value, abs_tol=1e-4):
                return True
        return False

    def index_of(self, value: float) -> int:
        for idx, pt in enumerate(self._pts):
            if math.isclose(pt, value, abs_tol=1e-4):
                return idx
        raise ValueError(f"value {value} not found")

    def min_max_perc(
        self, from_start: float = 0, to_end: float = 0
    ) -> tuple[float, float]:
        if from_start + to_end > 1.0:
            raise ValueError("requested range is empty")
        pt_min = self._start + from_start * self._length
        pt_max = self._end - to_end * self._length
        return pt_min, pt_max

    def min_max_n_pts(
        self, from_start: Union[int, float] = 0, to_end: Union[int, float] = 0
    ) -> tuple[float, float]:
        if isinstance(from_start, float):
            from_start = math.ceil(self._n_pts * from_start)
        if isinstance(to_end, float):
            to_end = math.floor(self._n_pts * to_end)

        pt_min = self._start + from_start * self._stepsize
        pt_max = self._end - to_end * self._stepsize
        if pt_min > pt_max:
            raise ValueError("requested range is empty")
        return pt_min, pt_max

    def sample_uniform(
        self, size: int = 1, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
        return rng.uniform(low=self._start, high=self._end, size=size)


class GridTime(Grid):
    def __init__(self, n_pts: int, stepsize: float, start=0.0):
        # NOTE:
        #   the zeroth timestep, containing in particular the
        #   initial-condition, should be handled separately by user
        # NOTE:
        #   add 1 to n-steps since we might want to track states before AND
        #   after time-stepping, which yields a total of (n-steps + 1) states
        #   in total
        super().__init__(n_pts=n_pts + 1, stepsize=stepsize, start=start)

        # (pre-)pad (just) enough zeros to make all timesteps uniform length
        self._formatter_timestep = f"{{:0{int(np.ceil(np.log10(self._n_pts)))}}}"

    @property
    def n_pts(self) -> int:
        return self._n_pts - 1

    def timestep_formatted(self, timestep: int) -> str:
        return f"{self._formatter_timestep}".format(timestep)

    def step(self, with_start: bool = False, with_end: bool = True) -> list[float]:
        return super().step(with_start, with_end)

    def step_with_index(
        self, with_start: bool = False, with_end: bool = True
    ) -> Iterable[tuple[int, float]]:
        return super().step_with_index(with_start, with_end)

    def is_init(self, val: float) -> bool:
        return math.isclose(val, self._start)


class Grids:
    def __init__(self, grids: list[Grid]):
        self._grids = grids
        self._n_dims = len(self._grids)
        self._dims = [gr.n_pts for gr in self._grids]

        self._starts, self._ends = (
            torch.tensor(list(self.starts()), dtype=torch.float),
            torch.tensor(list(self.ends()), dtype=torch.float),
        )

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def dims(self) -> list[int]:
        return self._dims

    def samples_sobol(self, n_samples: int, seed: Optional[int] = None) -> torch.Tensor:
        engine = torch.quasirandom.SobolEngine(self._n_dims, seed=seed)
        return engine.draw(n_samples) * (self._ends - self._starts) + self._starts

    def starts(self) -> Iterable[float]:
        return (gr.start for gr in self._grids)

    def ends(self) -> Iterable[float]:
        return (gr.end for gr in self._grids)

    def indexes(self) -> Generator[Iterable[int], None, None]:
        for idxs in itertools.product(*(range(gr.n_pts) for gr in self._grids)):
            yield idxs

    def steps(self) -> Generator[Iterable[float], None, None]:
        for vals in itertools.product(*(gr.step() for gr in self._grids)):
            yield vals

    def steps_with_index(self) -> Generator[Iterable[tuple[int, float]], None, None]:
        for idxs_vals in itertools.product(
            *(gr.step_with_index() for gr in self._grids)
        ):
            yield idxs_vals

    def boundaries(self) -> Generator[Iterable[float], None, None]:
        for vals in itertools.product(*(gr.step() for gr in self._grids)):
            if self.is_on_boundary(vals):
                yield vals

    def boundaries_with_index(
        self,
    ) -> Generator[Iterable[tuple[int, float]], None, None]:
        for idxs_vals in itertools.product(
            *(gr.step_with_index() for gr in self._grids)
        ):
            if self.is_on_boundary([val for idx, val in idxs_vals]):
                yield idxs_vals

    def internals(self) -> Generator[Iterable[float], None, None]:
        for vals in itertools.product(*(gr.step() for gr in self._grids)):
            if not self.is_on_boundary(vals):
                yield vals

    def internals_with_index(
        self,
    ) -> Generator[Iterable[tuple[int, float]], None, None]:
        for idxs_vals in itertools.product(
            *(gr.step_with_index() for gr in self._grids)
        ):
            if not self.is_on_boundary([val for idx, val in idxs_vals]):
                yield idxs_vals

    def is_on_boundary(self, vals: Iterable[float]) -> bool:
        for val, grid in zip(vals, self._grids):
            if grid.is_on_boundary(val):
                return True
        return False

    def zeroes_like(self) -> torch.Tensor:
        return torch.zeros([gr.n_pts for gr in self._grids])

    def constants_like(self, constant: float = 1.0) -> torch.Tensor:
        if math.isclose(constant, 0.0):
            return self.zeroes_like()
        return constant * torch.ones([gr.n_pts for gr in self._grids])

    def zeroes_like_numpy(self) -> torch.Tensor:
        return np.zeros([gr.n_pts for gr in self._grids])

    def coords_as_mesh(self, indexing_machine_like: bool = True) -> list[np.ndarray]:
        indexing: Literal["ij", "xy"]
        indexing = "ij" if indexing_machine_like else "xy"
        return np.meshgrid(*(gr.step() for gr in self._grids), indexing=indexing)

    def coords_as_mesh_torch(
        self, indexing_machine_like: bool = True
    ) -> list[torch.Tensor]:
        return [
            torch.from_numpy(mat) for mat in self.coords_as_mesh(indexing_machine_like)
        ]

    def sin_coords_as_mesh_torch(
        self, indexing_machine_like: bool = True
    ) -> list[torch.Tensor]:
        return [
            torch.sin(coord)
            for coord in self.coords_as_mesh_torch(indexing_machine_like)
        ]

    def cos_coords_as_mesh_torch(
        self, indexing_machine_like: bool = True
    ) -> list[torch.Tensor]:
        return [
            torch.cos(coord)
            for coord in self.coords_as_mesh_torch(indexing_machine_like)
        ]

    def flattten(self, target: torch.Tensor) -> torch.Tensor:
        res = []
        for indexes in self.indexes():
            res.append(target[indexes])
        return torch.tensor(res)

    def unflatten_2d(self, target: torch.Tensor) -> torch.Tensor:
        if len(self._grids) != 2:
            raise ValueError("expected 2d grid")
        n_gridpts = [gr.n_pts for gr in self._grids]
        res = torch.zeros(n_gridpts, dtype=target.dtype)
        for i, val in enumerate(target):
            row, col = divmod(i, n_gridpts[1])
            res[row, col] = val
        return res

    def mask(
        self, raw: torch.Tensor, idx_min: int, idx_max: int, val_mask: float = 0.0
    ) -> torch.Tensor:
        if raw.shape != torch.Size(self._dims):
            raise ValueError("wrong raw shape: must agree with that of the grids")

        res = self.constants_like(val_mask).to(raw.dtype)
        for indexes in self.indexes():
            if all([idx_min <= idx <= idx_max for idx in indexes]):
                res[indexes] = raw[indexes]
        return res

    def sample_uniform(
        self, size: int = 1, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
        return np.stack(
            [gr.sample_uniform(size=size, rng=rng) for gr in self._grids],
            axis=-1,  # such that first axis/index is size
        )
