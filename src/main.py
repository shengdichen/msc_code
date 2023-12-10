import logging

import torch

logger = logging.getLogger(__name__)


class Main:
    def __init__(self):
        self._report_lab_setup()

    @staticmethod
    def _report_lab_setup() -> None:
        if torch.cuda.is_available():
            logger.debug(f"Cuda'g on: {torch.cuda.get_device_name()}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s: [%(levelname)s] %(message)s", level=logging.DEBUG
    )
    Main()
