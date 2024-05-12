import logging
import pathlib
import random
import typing

import matplotlib.pyplot as plt
import src.pde.legacy.poisson.dataset
import torch
from src.deepl import factory
from src.definition import DEFINITION, T_NETWORK
from src.numerics import grid
from src.pde.poisson import dataset as dataset_poisson
from src.pde.poisson import learner as learner_poisson
from src.util import dataset as dataset_util

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self):
        self._grids = grid.Grids(
            [
                grid.Grid(n_pts=64, stepsize=0.01, start=0.0),
                grid.Grid(n_pts=64, stepsize=0.01, start=0.0),
            ]
        )

        self._n_instances_eval, self._n_instances_train = 300, 1800
        self._poisson_gauss = dataset_poisson.DatasetGauss(self._grids)
        self._poisson_sine = dataset_poisson.DatasetSin(self._grids)

        self._masks_eval = [
            dataset_util.MaskerRandom.from_min_max(
                intensity_min=i / 10, intensity_max=i / 10 + 0.1
            )
            for i in range(10)
        ]
        self._masks_eval_train = [
            self._masks_eval[0],
            self._masks_eval[3],
            self._masks_eval[6],
            self._masks_eval[9],
        ]
        self._masks_train = [
            dataset_util.MaskerRandom.from_min_max(
                intensity_min=0.0, intensity_max=1.0
            ),
            dataset_util.MaskerRandom.from_min_max(
                intensity_min=0.0, intensity_max=0.5
            ),
            dataset_util.MaskerRandom.from_min_max(
                intensity_min=0.5, intensity_max=1.0
            ),
            dataset_util.MaskerRandom.from_min_max(
                intensity_min=0.25, intensity_max=0.75
            ),
        ]

    def build(self) -> None:
        for dataset_raw in [self._poisson_gauss, self._poisson_sine]:
            src.pde.legacy.poisson.dataset.DatasetPoissonMaskedSolution.load_split(
                self._grids,
                dataset_raw,
                masks_eval=self._masks_eval,
                masks_train=self._masks_train,
                n_instances_eval=self._n_instances_eval,
                n_instances_train=self._n_instances_train,
                base_dir=DEFINITION.BIN_DIR / "poisson",
            )

    def train_adhoc(self) -> dict[str, T_NETWORK]:
        evals, trains = self._splits_mask_single(
            self._poisson_sine,
            masks_eval=self._masks_eval_train,
            masks_train=self._masks_train,
        )
        return self._train(trains, evals, use_adhoc_train=True)

    def train(self) -> dict[str, T_NETWORK]:
        evals, trains = self._splits_mask_single(
            self._poisson_sine,
            masks_eval=self._masks_eval_train,
            masks_train=self._masks_train,
        )
        return self._train(trains, evals, use_adhoc_train=False)

    def _train(
        self,
        trains: typing.Sequence[
            src.pde.legacy.poisson.dataset.DatasetPoissonMaskedSolution
        ],
        evals: typing.Sequence[
            src.pde.legacy.poisson.dataset.DatasetPoissonMaskedSolution
        ],
        use_adhoc_train: bool = False,
    ) -> dict[str, T_NETWORK]:
        datasets_eval = [ds.dataset for ds in evals]

        networks = {}
        for train in trains:
            for network, name_network in factory.Networks().networks(
                src.pde.legacy.poisson.dataset.DatasetPoissonMaskedSolution.N_CHANNELS_LHS,
                src.pde.legacy.poisson.dataset.DatasetPoissonMaskedSolution.N_CHANNELS_RHS,
            ):
                path = (
                    DEFINITION.BIN_DIR / "poisson" / f"{train.name}"
                    "--"
                    f"{name_network}"
                )
                if use_adhoc_train:
                    path = pathlib.Path(f"{str(path)}--adhoc")
                path = pathlib.Path(f"{str(path)}.pth")
                if not path.exists():
                    logger.info(f"training> {name_network} [{path}]")
                    learner = learner_poisson.LearnerPoissonFNOMaskedSolution(
                        self._grids, network
                    )
                    learner.train(
                        (
                            train.remake(self._n_instances_train)
                            if use_adhoc_train
                            else train.dataset
                        ),
                        n_epochs=1001,
                        batch_size=30,
                        datasets_eval=datasets_eval,
                    )
                    torch.save(network, path)

                networks[name_network] = torch.load(path)
        return networks

    def eval_poisson_single(self) -> None:
        dataset_raw = self._poisson_sine
        evals, trains = self._splits_mask_single(
            dataset_raw,
            masks_eval=self._masks_eval,
            masks_train=[dataset_util.MaskerRandom(i) for i in [0.5]],
        )
        datasets_eval = [ds.dataset for ds in evals]

        network_factory = factory.Networks()
        for train in trains:
            for __, name_network in network_factory.networks(
                src.pde.legacy.poisson.dataset.DatasetPoissonMaskedSolution.N_CHANNELS_LHS,
                src.pde.legacy.poisson.dataset.DatasetPoissonMaskedSolution.N_CHANNELS_RHS,
            ):
                path_base = (
                    DEFINITION.BIN_DIR / "poisson" / f"{train.name}"
                    "--"
                    f"{name_network}"
                )
                network = torch.load(pathlib.Path(f"{path_base}.pth"))
                logger.info(f"eval> {name_network}")

                learner = learner_poisson.LearnerPoissonFNOMaskedSolution(
                    self._grids, network
                )
                errors = [learner.errors(e)[0] for e in datasets_eval]
                print(errors)

                path_plot = pathlib.Path(f"{path_base}.png")
                if not path_plot.exists():
                    fig, ax = plt.subplots(dpi=200)
                    style = {"linestyle": "dashed", "linewidth": 1.5, "marker": "x"}
                    ax.plot(
                        [(i / 10) for i in range(10)],
                        errors,
                        **style,
                        label="random-style",
                    )
                    ax.set_xlabel("masking intensity (eval)")
                    ax.set_ylabel("error [$L^2$]")
                    ax.set_title(
                        f"Model: {name_network.upper()}\n"
                        f"Dataset: {train.name} [Training]"
                    )
                    ax.legend()
                    fig.savefig(path_plot)
                    plt.close(fig)

                for ev in evals:
                    eval_name, eval_ds = ev.name, ev.dataset
                    path_plot = pathlib.Path(
                        f"{path_base}--{eval_name}--compare-2d.png"
                    )
                    if not path_plot.exists():
                        fig = learner.plot_comparison_2d(eval_ds)
                        fig.savefig(path_plot)
                        plt.close(fig)
                    path_plot = pathlib.Path(
                        f"{path_base}--{eval_name}--compare-3d.png"
                    )
                    if not path_plot.exists():
                        fig = learner.plot_comparison_3d(eval_ds)
                        fig.savefig(path_plot)
                        plt.close(fig)

    def _splits_mask_single(
        self,
        dataset_raw: dataset_poisson.DatasetPoisson2d,
        masks_eval: typing.Iterable[dataset_util.Masker],
        masks_train: typing.Iterable[dataset_util.Masker],
    ) -> tuple[
        typing.Sequence[src.pde.legacy.poisson.dataset.DatasetPoissonMaskedSolution],
        typing.Sequence[src.pde.legacy.poisson.dataset.DatasetPoissonMaskedSolution],
    ]:
        return src.pde.legacy.poisson.dataset.DatasetPoissonMaskedSolution.load_split(
            self._grids,
            dataset_raw,
            masks_eval=masks_eval,
            masks_train=masks_train,
            n_instances_eval=self._n_instances_eval,
            n_instances_train=self._n_instances_train,
            base_dir=DEFINITION.BIN_DIR / "poisson",
        )


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)

    pipeline = Pipeline()
    pipeline.build()
    # pipeline.train()
    pipeline.train_adhoc()
    # pipeline.eval_poisson_single()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s [%(levelname)s]> %(message)s", level=logging.INFO
    )
    main()
