import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


class Multidiff:
    def __init__(self, rhs: torch.Tensor, lhs: torch.Tensor):
        self._rhs, self._lhs = self._preprocess(rhs, lhs)

    def _preprocess(
        self, rhs: torch.Tensor, lhs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_dims_rhs = len(rhs.shape)
        if n_dims_rhs > 2:
            raise ValueError("max allowed dimension of rhs is 2")
        if n_dims_rhs == 2:
            if rhs.shape[1] > 1:
                raise ValueError("second dimension must be 1")
        if n_dims_rhs <= 1:
            rhs = rhs.view(-1, 1)

        n_dims_lhs = len(lhs.shape)
        if n_dims_lhs > 2:
            raise ValueError("max allowed dimension of lhs is 2")
        if n_dims_lhs <= 1:
            lhs = lhs.view(-1, 1)
        if not lhs.requires_grad:
            logger.warning(
                f"{self.__class__}> lhs has |requires_grad| set to FALSE, "
                f"setting to TRUE; but will always return zeroes when differentiating"
            )
            lhs.requires_grad = True

        if len(lhs) != len(rhs):
            raise ValueError("lhs and rhs must be of same length (first dimension)")
        return rhs, lhs

    @property
    def rhs(self) -> torch.Tensor:
        return self._rhs

    def _rhs_to_scalar(self) -> None:
        if self._rhs.shape:
            rhs_next = torch.sum(self._rhs)
            self._rhs = rhs_next

    def diff(self) -> torch.Tensor:
        self._rhs_to_scalar()

        if not self._rhs.grad_fn:
            logger.warning(
                "rhs has no |grad_fn|, returning zeroes "
                "(you should probably stop differentiating now)"
            )
            result = torch.zeros_like(self._lhs)
            self._rhs = torch.zeros_like(self._rhs)
        else:
            result = torch.autograd.grad(
                self._rhs,
                self._lhs,
                create_graph=True,  # so we could keep diff'g
            )[0]
            self._rhs = result

        logger.debug(f"rhs [{self._rhs}] with size [{self._rhs.shape}]")
        return result


class MultidiffNetwork:
    def __init__(
        self,
        network: torch.nn.Module,
        lhs: torch.Tensor,
        lhs_names: Optional[list[str]] = None,
    ):
        self._network = network
        self._lhs, self._lhs_names = self._process_lhs(lhs), lhs_names
        if self._lhs_names:
            if self._lhs.shape[1] != len(self._lhs_names):
                raise ValueError("lhs and lhs_names must have equal length")

        self._diff_0 = self._network(self._lhs)
        self._diffs: list[list[torch.Tensor]] = []
        for idx in range(lhs.shape[1]):
            self._diffs.append(
                [self._tensor_at_index(Multidiff(self._diff_0, self._lhs).diff(), idx)]
            )

    def _process_lhs(self, lhs: torch.Tensor) -> torch.Tensor:
        n_dims = len(lhs.shape)
        if n_dims < 1 or n_dims > 2:
            raise ValueError

        if not lhs.requires_grad:
            lhs.requires_grad = True
        if n_dims == 1:
            return lhs.view(-1, 1)
        return lhs

    def _tensor_at_index(self, target: torch.Tensor, index: int) -> torch.Tensor:
        return target[
            :,
            index : index + 1,  # so that we do not lose a dimension
        ]

    def _name_to_index(self, name: str) -> int:
        if not self._lhs_names:
            raise ValueError("attempting to index lhs by names, but they are unnamed")
        return self._lhs_names.index(name)

    def diff_0(self) -> torch.Tensor:
        return self._diff_0

    def diff(self, target: Union[int, str], order: int) -> torch.Tensor:
        if order == 0:
            raise ValueError(
                "Call <object>.diff_0() directly to evaluate network at lhs"
            )

        if isinstance(target, str):
            target = self._name_to_index(target)
        diffs = self._diffs[target]
        if order > len(diffs):
            self._build_pool(target, order)
        return diffs[order - 1]  # first entry (at index 0) is diff-order-1

    def _build_pool(self, index: int, order: int) -> None:
        diffs = self._diffs[index]
        for od in range(len(diffs) + 1, order + 1):
            logger.debug(f"building [{od}]th derivative of [{index}]th lhs")
            diffs.append(
                self._tensor_at_index(Multidiff(diffs[-1], self._lhs).diff(), index)
            )
