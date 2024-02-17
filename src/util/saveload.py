import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union

import torch
from matplotlib import figure

from src.definition import DEFINITION

logger = logging.getLogger(__name__)


class Saveloader:
    def __init__(self, base: Path = DEFINITION.BIN_DIR):
        self._base = base

    @property
    def base(self) -> Path:
        return self._base

    def rebase_location(self, location: Union[Path, str]) -> Path:
        return Path(self._base, location)

    def exists(self, location: Path) -> bool:
        return location.exists()

    def load(self, location: Path) -> Any:
        if not self.exists(location):
            raise FileNotFoundError
        logger.info(f"target found at {location}")
        return self._load(location)

    def _load(self, location: Path) -> Any:
        pass

    def save(self, target: Any, location: Path, overwrite: bool = False) -> None:
        if self.exists(location):
            if not overwrite:
                raise FileExistsError(f"{location} already exists")
            location.unlink()
        else:
            logger.info(f"target saved at {location}")
            self._make_folder_containing(location)
            self._save(target, location)

    def _save(self, target: Any, location: Path) -> None:
        pass

    def _make_folder_containing(self, location: Path) -> None:
        location.parent.mkdir(parents=True, exist_ok=True)

    def load_or_make(self, location: Path, make_target: Callable[[], Any]) -> Any:
        if self.exists(location):
            return self.load(location)

        logger.info(f"target NOT found at {location}; making it now")
        target = make_target()
        self.save(target, location)
        return target


class SaveloadImage(Saveloader):
    def __init__(self, folder: Union[Path, str], suffix: str = "png"):
        super().__init__(Path(DEFINITION.BIN_DIR, folder))

        self._suffix = suffix

    def rebase_location(self, location: Union[Path, str]) -> Path:
        return Path(self._base, f"{location}.{self._suffix}")

    def _save(self, target: figure.FigureBase, location: Path) -> None:
        target.savefig(location)


class SaveloadTorch(Saveloader):
    def __init__(self, folder: Union[Path, str], torch_suffix: str = "pth"):
        super().__init__(Path(DEFINITION.BIN_DIR, folder))

        self._torch_suffix = torch_suffix

    def rebase_location(self, location: Union[Path, str]) -> Path:
        return Path(self._base, f"{location}.{self._torch_suffix}")

    def _load(self, location: Path) -> None:
        return torch.load(location)

    def _save(self, target: Any, location: Path) -> None:
        torch.save(target, location)


class SaveloadPde(SaveloadTorch):
    def __init__(
        self,
        folder: Union[Path, str],
        torch_suffix: str = "pth",
        name_init: str = "init",
        name_boundary: str = "boundary",
        name_internal: str = "internal",
    ):
        super().__init__(folder, torch_suffix)

        self._folder = folder
        os.makedirs(self._folder, exist_ok=True)
        self._torch_suffix = torch_suffix

        self._mode_to_path: dict[str, Path] = {
            mode: self.rebase_location(name)
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
