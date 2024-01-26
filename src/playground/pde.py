import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Generate synthetic data for the direct problem (forward model)
def generate_y_heat(x):
    # 1d heat equation
    return np.sin(np.pi * x)


x_train = np.random.rand(100, 1)
y_train = generate_y_heat(x_train)


class InverseModel(nn.Module):
    def __init__(self):
        super(InverseModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = InverseModel()


def train() -> None:
    lr = 0.01
    n_epochs = 1000

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)

    for ep in range(n_epochs):
        # Forward pass
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 100 == 0:
            print(f"Epoch [{ep+1}/{n_epochs}], Loss: {loss.item():.4f}")


def test() -> None:
    x_test = np.linspace(0, 1, 100).reshape(-1, 1)
    x_test_tensor = torch.FloatTensor(x_test)
    with torch.no_grad():
        y_pred = model(x_test_tensor)

    plt.scatter(x_train, y_train, label="Training Data")
    plt.plot(x_test, y_pred.numpy(), label="Inverse Model Prediction", color="r")
    plt.xlabel("Spatial Location")
    plt.ylabel("Temperature")
    plt.legend()
    plt.savefig("fig.png")
    # plt.show()


if __name__ == "__main__":
    train()
    test()
