import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from dill.tests.test_classdef import o
from torch.utils.data import DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def exact_solution(x):
    return torch.sin(x)


# Type of optimizer (ADAM or LBFGS)
opt_type = "ADAM"
# Random Seed for dataset generation
sampling_seed = 78
torch.manual_seed(sampling_seed)

# Number of training samples
n_samples = 20
# Noise level
sigma = 0.25

x = 2 * np.pi * torch.rand((n_samples, 1))
y = exact_solution(x) + sigma * torch.randn(x.shape)

batch_size = 5
training_set = DataLoader(
    torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=True
)

plt.grid(True, which="both", ls=":")
plt.scatter(x.detach(), y)
plt.xlabel("x")
plt.ylabel("u")


class NeuralNet(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons):
        super(NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons or units per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = nn.Tanh()

        if self.n_hidden_layers != 0:
            self.input_layer = nn.Linear(self.input_dimension, self.neurons)
            self.hidden_layers = nn.ModuleList(
                [
                    nn.Linear(self.neurons, self.neurons)
                    for _ in range(n_hidden_layers - 1)
                ]
            )
            self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        else:
            print("Simple linear regression")
            self.linear_regression_layer = nn.Linear(
                self.input_dimension, self.output_dimension
            )

    def forward(self, x):
        # The forward function performs the set of affine and
        # non-linear transformations defining the network (see equation above).
        if self.n_hidden_layers != 0:
            x = self.activation(self.input_layer(x))
            for k, l in enumerate(self.hidden_layers):
                x = self.activation(l(x))
            return self.output_layer(x)
        else:
            return self.linear_regression_layer(x)


def NeuralNet_Seq(input_dimension, output_dimension, n_hidden_layers, neurons):
    modules = list()
    modules.append(nn.Linear(input_dimension, neurons))
    modules.append(nn.Tanh())
    for _ in range(n_hidden_layers):
        modules.append(nn.Linear(neurons, neurons))
        modules.append(nn.Tanh())
    modules.append(nn.Linear(neurons, output_dimension))
    model = nn.Sequential(*modules)
    return model


# Model definition
my_network = NeuralNet(
    input_dimension=x.shape[1],
    output_dimension=y.shape[1],
    n_hidden_layers=2,
    neurons=20,
)


# u_pred = my_network(x)
# u_ex = exact_solution(x)
# print(u_pred, u_ex)

# my_network = NeuralNet_Seq(input_dimension=x.shape[1], output_dimension=y.shape[1], n_hidden_layers=1, neurons=20)


def init_xavier(model, retrain_seed):
    torch.manual_seed(retrain_seed)

    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain("tanh")
            torch.nn.init.xavier_uniform_(m.weight, gain=g)
            # torch.nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)

    model.apply(init_weights)


# Random Seed for weight initialization
retrain = 1456
# Xavier weight initialization
init_xavier(my_network, retrain)
# Model definition

# Predict network value of x
# print(my_network(x))
