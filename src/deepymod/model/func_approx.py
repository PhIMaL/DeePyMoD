import torch
import torch.nn as nn
from typing import List
import numpy as np


class NN(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, n_in: int, n_hidden: List[int], n_out: int) -> None:
        super().__init__()
        self.network = self.build_network(n_in, n_hidden, n_out)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            input (torch.Tensor): [description]

        Returns:
            torch.Tensor: [description]
        """
        coordinates = input.clone().detach().requires_grad_(True)
        return self.network(coordinates), coordinates

    def build_network(self, n_in: int, n_hidden: List[int], n_out: int) -> torch.nn.Sequential:
        """ Constructs a feed-forward neural network.

        Args:
            n_in (int): Number of input features.
            n_hidden (list[int]): Number of neurons in each layer.
            n_out (int): Number of output features.

        Returns:
            torch.Sequential: Pytorch module
        """

        network = []
        architecture = [n_in] + n_hidden + [n_out]
        for layer_i, layer_j in zip(architecture, architecture[1:]):
            network.append(nn.Linear(layer_i, layer_j))
            network.append(nn.Tanh())
        network.pop()  # get rid of last activation function
        return nn.Sequential(*network)


class SineLayer(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, in_features: int, out_features: int, omega_0: float = 30, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)

        self.init_weights()

    def init_weights(self):
        """[summary]
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        """[summary]

        Args:
            input ([type]): [description]

        Returns:
            [type]: [description]
        """
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, n_in: int, n_hidden: List[int], n_out: int, first_omega_0: float = 30., hidden_omega_0: float = 30.):
        super().__init__()
        self.network = self.build_network(n_in, n_hidden, n_out, first_omega_0, hidden_omega_0)

    def forward(self, input: torch.Tensor):
        """[summary]

        Args:
            input (torch.Tensor): [description]

        Returns:
            [type]: [description]
        """
        coordinates = input.clone().detach().requires_grad_(True)
        return self.network(coordinates), coordinates

    def build_network(self, n_in: int, n_hidden: List[int], n_out: int, first_omega_0: float, hidden_omega_0: float):
        """[summary]

        Args:
            n_in (int): [description]
            n_hidden (List[int]): [description]
            n_out (int): [description]
            first_omega_0 (float): [description]
            hidden_omega_0 (float): [description]

        Returns:
            [type]: [description]
        """
        network = []
        # Input layer
        network.append(SineLayer(n_in, n_hidden[0], is_first=True, omega_0=first_omega_0))

        # Hidden layers
        for layer_i, layer_j in zip(n_hidden, n_hidden[1:]):
            network.append(SineLayer(layer_i, layer_j, is_first=False, omega_0=hidden_omega_0))

        # Output layer
        final_linear = nn.Linear(n_hidden[-1], n_out)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / n_hidden[-1]) / hidden_omega_0,
                                         np.sqrt(6 / n_hidden[-1]) / hidden_omega_0)
            network.append(final_linear)

        return nn.Sequential(*network)
