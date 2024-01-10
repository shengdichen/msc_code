import torch


class Network(torch.nn.Module):
    def __init__(
        self,
        dim_x: int,
        with_time: bool = True,
        size_hidden: int = 30,
        n_hiddens: int = 2,
        activation=torch.nn.Tanh,
    ):
        super().__init__()

        self._size_input = dim_x
        if with_time:
            self._size_input += 1

        self._size_hidden, self._n_hiddens = size_hidden, n_hiddens
        self._activation = torch.nn.Tanh

        self._layers = self._make_layers()

    def _make_layers(self) -> torch.nn.Module:
        layers = []

        layers.append(torch.nn.Linear(self._size_input, self._size_hidden))
        for _ in range(self._n_hiddens):
            layers.append(torch.nn.Linear(self._size_hidden, self._size_hidden))
        layers.append(
            torch.nn.Linear(
                self._size_hidden,
                1,  # always outputs to R^1
            )
        )

        return torch.nn.ModuleList(layers)

    def forward(self, lhs: torch.Tensor) -> torch.Tensor:
        res = self._activation()(self._layers[0](lhs))
        for layer in self._layers[1:-1]:
            res = self._activation()(layer(res))
        return self._layers[-1](res)
