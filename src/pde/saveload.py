import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class Saveload:
    def __init__(
        self,
        folder: Path,
        torch_suffix: str = "pth",
        name_init: str = "init",
        name_boundary: str = "boundary",
        name_internal: str = "internal",
    ):
        self._folder = folder
        os.makedirs(self._folder, exist_ok=True)
        self._torch_suffix = torch_suffix

        self._mode_to_path: dict[str, Path] = {
            mode: self._folder / f"{name}.{self._torch_suffix}"
            for mode, name in zip(
                ["init", "boundary", "internal"],
                [name_init, name_boundary, name_internal],
            )
        }

    def dataset_init(
        self, lhss: list[torch.Tensor], rhss: list[float]
    ) -> torch.utils.data.dataset.TensorDataset:
        return self._dataset_saveload("init", lhss, rhss)

    def dataset_boundary(
        self, lhss: list[torch.Tensor], rhss: list[float]
    ) -> torch.utils.data.dataset.TensorDataset:
        return self._dataset_saveload("boundary", lhss, rhss)

    def dataset_internal(
        self, lhss: list[torch.Tensor], rhss: list[float]
    ) -> torch.utils.data.dataset.TensorDataset:
        return self._dataset_saveload("internal", lhss, rhss)

    def exists_init(self) -> bool:
        return self._mode_to_path["init"].exists()

    def exists_boundary(self) -> bool:
        return self._mode_to_path["boundary"].exists()

    def exists_internal(self) -> bool:
        return self._mode_to_path["internal"].exists()

    def _dataset_saveload(
        self, mode: str, lhss: torch.Tensor, rhss: list[float]
    ) -> torch.utils.data.dataset.TensorDataset:
        path = self._mode_to_path[mode]

        if not path.exists():
            dataset = self._convert_to_dataset(lhss, rhss)
            torch.save(dataset, path)
            logger.info(f"dataset ({mode}) saved at: {path}")
            return dataset

        logger.info(f"dataset ({mode}) found at: {path}")
        return torch.load(path)

    def _convert_to_dataset(
        self, lhss: list[torch.Tensor], rhss: list[float]
    ) -> torch.utils.data.dataset.TensorDataset:
        return torch.utils.data.TensorDataset(
            torch.stack(lhss), torch.tensor(rhss).view(-1, 1)
        )
