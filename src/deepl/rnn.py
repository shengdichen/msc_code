import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self._hidden_size = hidden_size
        self._layer_rnn = nn.RNN(input_size, self._hidden_size, batch_first=True)
        self._layer_fc = nn.Linear(self._hidden_size, output_size)

    def forward(self, x, hidden):
        out_rnn, hidden = self._layer_rnn(x, hidden)
        out = self._layer_fc(out_rnn[:, -1, :])  # Taking the last time step's output
        return out, hidden

    def init_hidden(self, batch_size) -> None:
        return torch.zeros(1, batch_size, self._hidden_size)


hyperparam = {
    "input_size": 1,
    "hidden_size": 32,
    "output_size": 1,
    "seq_length": 20,
    "batch_size": 10,
    "epochs": 100,
    "lr": 0.01,
}


def input_seq() -> torch.Tensor:
    return torch.randn(
        hyperparam["batch_size"], hyperparam["seq_length"], hyperparam["input_size"]
    )


def target_seq() -> torch.Tensor:
    return torch.randn(hyperparam["batch_size"], hyperparam["output_size"])


class Model:
    def __init__(self):
        self._model = SimpleRNN(
            hyperparam["input_size"],
            hyperparam["hidden_size"],
            hyperparam["output_size"],
        )
        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=hyperparam["lr"]
        )

    def train(self) -> None:
        for epoch in range(int(hyperparam["epochs"])):
            self._optimizer.zero_grad()
            hidden = self._model.init_hidden(hyperparam["batch_size"])

            output, _ = self._model(input_seq(), hidden)
            loss = self._criterion(output, target_seq())

            loss.backward()
            self._optimizer.step()

            if (epoch + 1) % 10 == 0:
                epochs = {hyperparam["epochs"]}
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        print("Training complete!")

    def test(self) -> None:
        # Test the model with a sample input sequence
        test_input = torch.randn(
            1, hyperparam["seq_length"], hyperparam["input_size"]
        )  # Single batch
        with torch.no_grad():
            hidden = self._model.init_hidden(1)
            pred, _ = self._model(test_input, hidden)

        print("Predicted next value:", pred.item())


if __name__ == "__main__":
    m = Model()
    m.train()
    m.test()
