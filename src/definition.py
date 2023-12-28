from pathlib import Path


class Definition:
    SRC_DIR = Path(__file__).parent

    ROOT_DIR = SRC_DIR.parent
    TEST_DIR = SRC_DIR.parent / "tests"

    BIN_DIR = SRC_DIR.parent / "bin"


DEFINITION = Definition()
