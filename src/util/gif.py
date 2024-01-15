import fnmatch
from pathlib import Path

import imageio.v3 as imageio_v3


class MakerGif:
    def __init__(self, source_dir: Path, source_suffix: str = "png"):
        self._source_dir = source_dir
        self._suffix = source_suffix

    def make(self, out: Path) -> None:
        frames = []
        for f in self._source_dir.iterdir():
            if f.is_file() and fnmatch.fnmatch(f.name, f"*.{self._suffix}"):
                frames.append(f)
        frames.sort()

        imageio_v3.imwrite(out, [imageio_v3.imread(f) for f in frames])
