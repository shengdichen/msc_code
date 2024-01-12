import math
from typing import Any

import torch


class EqualityFloatlike:
    def __init__(self, ours: Any, theirs: Any, tolerance=1e-4):
        self._ours, self._theirs = ours, theirs
        self._tolerance = tolerance

    def is_close(self) -> bool:
        raise NotImplementedError

    def is_equal(self) -> bool:
        raise NotImplementedError


class EqualityBuiltin(EqualityFloatlike):
    def __init__(self, ours: list, theirs: list, *args, **kwargs):
        super().__init__(ours, theirs, *args, **kwargs)

    def _equal_lengths(self) -> bool:
        return len(self._ours) == len(self._theirs)

    def is_close(self) -> bool:
        return self._equal_lengths() and all(
            (
                math.isclose(ours, theirs, abs_tol=self._tolerance)
                for ours, theirs in zip(self._ours, self._theirs)
            )
        )

    def is_equal(self) -> bool:
        return self._equal_lengths() and all(
            (
                math.isclose(ours, theirs)
                for ours, theirs in zip(self._ours, self._theirs)
            )
        )


class EqualityTorch(EqualityFloatlike):
    def __init__(self, ours: torch.Tensor, theirs: torch.Tensor, *args, **kwargs):
        super().__init__(ours, theirs, *args, **kwargs)

    def is_close(self) -> bool:
        return torch.all(torch.abs(self._ours - self._theirs) < self._tolerance)

    def is_equal(self) -> bool:
        return torch.equal(self._ours, self._theirs)
