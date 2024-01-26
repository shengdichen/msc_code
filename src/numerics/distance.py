from typing import Union

import torch


class Distance:
    def __init__(
        self,
        ours: Union[torch.Tensor, float, int],
        theirs: Union[torch.Tensor, float, int] = 0.0,
    ):
        if not isinstance(ours, torch.Tensor):
            if not isinstance(ours, float):
                ours = float(ours)
            ours = torch.tensor(ours)
        self._ours = ours

        if isinstance(theirs, int):
            theirs = float(theirs)
        self._theirs = theirs

    def mse(self) -> torch.Tensor:
        return torch.mean((self._ours - self._theirs) ** 2)

    def mse_relative(self) -> torch.Tensor:
        return (self.mse() / torch.mean(self._theirs**2)) ** 0.5

    def mse_percentage(self, precision: int = 4) -> str:
        return f"{self.mse_relative().item():.{precision}%}"

    def norm_lp(self, p: int) -> torch.Tensor:
        if p % 2:
            diffs = torch.abs(self._ours - self._theirs)
        else:
            diffs = self._ours - self._theirs

        return torch.sum(diffs**p) ** (1 / p)
