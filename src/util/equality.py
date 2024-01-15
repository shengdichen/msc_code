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


class EqualityTorch(EqualityFloatlike):
    def __init__(self, ours: torch.Tensor, theirs: torch.Tensor, *args, **kwargs):
        super().__init__(ours, theirs, *args, **kwargs)

    def is_close(self) -> bool:
        return torch.all(torch.abs(self._ours - self._theirs) < self._tolerance)

    def is_equal(self) -> bool:
        return torch.equal(self._ours, self._theirs)
