import logging

import torch

logger = logging.getLogger(__name__)


class Multidiff:
    def __init__(
        self, rhs: torch.Tensor, lhs: torch.Tensor, force_rhs_scalar: bool = False
    ):
        self._force_rhs_scalar = force_rhs_scalar
        self._rhs, self._lhs = self._preprocess(rhs, lhs)
        self._ones = torch.tensor(1)

    def _preprocess(
        self, rhs: torch.Tensor, lhs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rhs.shape:
            if self._force_rhs_scalar:
                rhs = torch.squeeze(rhs)
                rhs_shape = rhs.shape
                if rhs_shape:
                    raise ValueError(
                        f"{self.__class__}> expected output without shape, "
                        f"but got (post-squeezing) [{rhs}] with shape [{rhs_shape}]"
                    )
            else:
                rhs = torch.sum(rhs)

        if not lhs.requires_grad:
            logger.warning(
                f"{self.__class__}> lhs has |requires_grad| set to FALSE, "
                f"setting to TRUE; but will always return zeroes when differentiating"
            )
            lhs.requires_grad = True

        return rhs, lhs

    @property
    def rhs(self) -> torch.Tensor:
        return self._rhs

    def _rhs_to_scalar(self) -> None:
        if self._rhs.shape:
            rhs_next = torch.sum(self._rhs)
            self._rhs = rhs_next

    def diff(self) -> torch.Tensor:
        if not self._rhs.grad_fn:
            logger.warning(
                "rhs has no |grad_fn|, returning zeroes "
                "(you should probably stop differentiating now)"
            )
            result = torch.zeros_like(self._lhs)
            self._rhs = torch.tensor(0)
        else:
            result = torch.autograd.grad(
                self._rhs,
                self._lhs,
                grad_outputs=self._ones,
                create_graph=True,  # so we could keep diff'g
            )[0]
            self._rhs = result
            self._rhs_to_scalar()  # prepare for next diff()-call

        logger.debug(f"rhs [{self._rhs}] with size [{self._rhs.shape}]")
        return result
