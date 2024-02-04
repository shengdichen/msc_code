from pathlib import Path

import torch


class Definition:
    SRC_DIR = Path(__file__).parent

    ROOT_DIR = SRC_DIR.parent
    TEST_DIR = SRC_DIR.parent / "tests"

    BIN_DIR = SRC_DIR.parent / "bin"

    def __init__(self):
        self._has_cuda = torch.cuda.is_available()

    @property
    def has_cuda(self) -> bool:
        return self._has_cuda

    @property
    def device_preferred(self) -> str:
        return "cuda" if self._has_cuda else "cpu"


DEFINITION = Definition()
