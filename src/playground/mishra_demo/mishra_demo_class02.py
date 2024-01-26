import matplotlib.pyplot as plt
import numpy as np


class RawNN:
    def __init__(self):
        np.random.seed(123)

        start, end = -1, 2
        self._x = np.linspace(start, end, 40).reshape((-1, 1, 1))
        self._x_true = np.linspace(start, end, 100).reshape((-1, 1, 1))

        self._y = self._make_y(self._x, add_noise=True)
        self._y_true = self._make_y(self._x_true, add_noise=False)

    @staticmethod
    def _make_y(x: np.ndarray, add_noise: bool) -> np.ndarray:
        theta = [0, -1, -3, 2]
        y = theta[3] * x**3 + theta[2] * x**2 + theta[1] * x + theta[0]
        if add_noise:
            y += np.random.normal(0, 0.15, x.shape)

        return y

    @staticmethod
    def activate_tanh(input: np.ndarray) -> np.ndarray:
        return np.tanh(input)

    @staticmethod
    def activate_deriv_tanh(input: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(input) ** 2

    def visualize_data(self) -> None:
        plt.figure()
        plt.plot(self._x_true[:, 0, 0], self._y_true[:, 0, 0], label="Exact function")
        plt.scatter(self._x[:, 0, 0], self._y[:, 0, 0], label="Noisy training data")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.legend()
        plt.savefig("file")

    def train(self) -> None:
        np.random.seed(0)
        size_layer_1, size_layer_2 = 8, 16

        W1 = np.random.uniform(-1, 1, (1, size_layer_1, 1))
        W2 = np.random.uniform(-1, 1, (1, size_layer_2, size_layer_1)) * np.sqrt(1 / 8)
        W3 = np.random.uniform(-1, 1, (1, 1, size_layer_2)) * np.sqrt(1 / 16)

        b1 = np.random.uniform(-1, 1, (1, size_layer_1, 1))
        b2 = np.random.uniform(-1, 1, (1, size_layer_2, 1)) * np.sqrt(1 / 8)
        b3 = np.random.uniform(-1, 1, (1, 1, 1)) * np.sqrt(1 / 16)

        for i in range(2001):
            # forward pass
            h = W1 @ self._x + b1  # h: (N_examples, 8, 1)
            g = W2 @ self.activate_tanh(h) + b2  # g: (N_examples, 16, 1)
            f = W3 @ self.activate_tanh(g) + b3  # f: (N_examples, 1, 1)

            # reverse-mode backpropagation
            dldf = 2 * (f - self._y)  # dl/df: (N_examples, 1, 1)
            dldg = (dldf @ W3) @ (
                self.activate_deriv_tanh(g) * np.expand_dims(np.eye(16), 0)
            )  # dl/dg: (N_examples, 1, 16)
            dldh = (dldg @ W2) @ (
                self.activate_deriv_tanh(h) * np.expand_dims(np.eye(8), 0)
            )  # dl/dh: (N_examples, 1, 8)

            dW1 = (dldh * self._x).transpose(0, 2, 1)  # dl/dW1: (N_examples, 8, 1)
            dW2 = (dldg * self.activate_tanh(h)).transpose(
                0, 2, 1
            )  # dl/dW2: (N_examples, 16, 8)
            dW3 = (dldf * self.activate_tanh(g)).transpose(
                0, 2, 1
            )  # dl/dW3: (N_examples, 1, 16)

            db1 = dldh.transpose(0, 2, 1)  # dl/db1: (N_examples, 8, 1)
            db2 = dldg.transpose(0, 2, 1)  # dl/db2: (N_examples, 16, 1)
            db3 = dldf.transpose(0, 2, 1)  # dl/db3: (N_examples, 1, 1)

            # get mean gradient across training examples
            dW1 = np.mean(dW1, 0, keepdims=True)
            dW2 = np.mean(dW2, 0, keepdims=True)
            dW3 = np.mean(dW3, 0, keepdims=True)
            db1 = np.mean(db1, 0, keepdims=True)
            db2 = np.mean(db2, 0, keepdims=True)
            db3 = np.mean(db3, 0, keepdims=True)

            # gradient descent step
            a = 0.05  # learning rate
            W1 -= a * dW1
            W2 -= a * dW2
            W3 -= a * dW3
            b1 -= a * db1
            b2 -= a * db2
            b3 -= a * db3

            if i == 0:
                print(h.shape, g.shape, f.shape)
                print(dldh.shape, dldg.shape, dldf.shape)
                print(dW1.shape, dW2.shape, dW3.shape)
                print(db1.shape, db2.shape, db3.shape)

            if i % 500 == 0:
                # get loss value
                l = np.mean((f - self._y) ** 2)

                # Plot the training data
                plt.figure()
                plt.title(f"step: {i}, training loss: {l:.2f}")
                plt.plot(
                    self._x_true[:, 0, 0], self._y_true[:, 0, 0], label="Exact function"
                )
                plt.scatter(
                    self._x[:, 0, 0], self._y[:, 0, 0], label="Noisy training data"
                )
                plt.plot(self._x[:, 0, 0], f[:, 0, 0], lw=3, label="Neural network")
                plt.xlabel("$x$")
                plt.ylabel("$y$")
                plt.legend()
                plt.show()


if __name__ == "__main__":
    nn = RawNN()
    nn.visualize_data()
