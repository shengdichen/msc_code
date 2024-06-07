import random
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch

T_DATASET = torch.utils.data.dataset.TensorDataset
T_NETWORK = torch.nn.Module


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

    @staticmethod
    def seed(seed: int = 42) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def configure_font_matplotlib(font_latex: bool = True) -> None:
        mpl.rcParams["font.family"] = "serif"

        if font_latex:
            mpl.rcParams["text.usetex"] = True
            mpl.rcParams["mathtext.fontset"] = "custom"
            mpl.rcParams["text.latex.preamble"] = (
                "\\usepackage{libertine,libertinust1math}"
            )
        else:
            mpl.rcParams["font.sans-serif"].insert(0, "Avenir LT Std")
            mpl.rcParams["font.serif"].insert(0, "Libertinus Serif")


DEFINITION = Definition()
