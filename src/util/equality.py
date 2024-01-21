import math
from collections.abc import Iterable
from typing import Any, Union

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
    def __init__(
        self,
        ours: Union[float, Iterable[float]],
        theirs: Union[float, Iterable[float]],
        *args,
        **kwargs
    ):
        if not isinstance(ours, Iterable):
            ours = [ours]
        if not isinstance(theirs, Iterable):
            theirs = [theirs]

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

    def _equal_shape(self) -> bool:
        return self._ours.shape == self._theirs.shape

    def is_close(self) -> bool:
        return self._equal_shape() and torch.all(
            torch.abs(self._ours - self._theirs) < self._tolerance
        )

    def is_equal(self) -> bool:
        return self._equal_shape() and torch.equal(self._ours, self._theirs)
