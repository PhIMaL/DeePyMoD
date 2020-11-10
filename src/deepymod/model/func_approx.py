""" This files contains the function approximators that are used by DeepMoD.
"""


import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np


class NN(nn.Module):
    def __init__(self, n_in: int, n_hidden: List[int], n_out: int) -> None:
        """ Constructs a feed-forward neural network with tanh activation.

        Args:
            n_in (int): Number of input features.
            n_hidden (List[int]): Number of neurons in each layer.
            n_out (int): Number of output features.
        """
        super().__init__()
        self.network = self.build_network(n_in, n_hidden, n_out)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network. Returns prediction and the differentiable input
        so we can construct the library.

        Args:
            input (torch.Tensor): Input tensor of size (n_samples, n_inputs).

        Returns:
            (torch.Tensor, torch.Tensor): prediction of size (n_samples, n_outputs) and coordinates of size (n_samples, n_inputs).
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
    def __init__(self, in_features: int, out_features: int, omega_0: float = 30, is_first: bool = False) -> None:
        """ Sine activation function layer with omega_0 scaling.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            omega_0 (float, optional): Scaling factor of the Sine function. Defaults to 30.
            is_first (bool, optional): Defaults to False.
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialization of the weigths."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer.

        Args:
            input (torch.Tensor): Input tensor of shape (n_samples, n_inputs).

        Returns:
            torch.Tensor: Prediction of shape (n_samples, n_outputs)
        """
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, n_in: int, n_hidden: List[int], n_out: int, first_omega_0: float = 30., hidden_omega_0: float = 30.) -> None:
        """ SIREN model from the paper [Implicit Neural Representations with
        Periodic Activation Functions](https://arxiv.org/abs/2006.09661).

        Args:
            n_in (int): Number of input features.
            n_hidden (list[int]): Number of neurons in each layer.
            n_out (int): Number of output features.
            first_omega_0 (float, optional): Scaling factor of the Sine function of the first layer. Defaults to 30.
            hidden_omega_0 (float, optional): Scaling factor of the Sine function of the hidden layers. Defaults to 30.
        """
        super().__init__()
        self.network = self.build_network(n_in, n_hidden, n_out, first_omega_0, hidden_omega_0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            input (torch.Tensor): Input tensor of shape (n_samples, n_inputs).

        Returns:
            torch.Tensor: Prediction of shape (n_samples, n_outputs)
        """
        coordinates = input.clone().detach().requires_grad_(True)
        return self.network(coordinates), coordinates

    def build_network(self, n_in: int, n_hidden: List[int], n_out: int, first_omega_0: float, hidden_omega_0: float) -> torch.nn.Sequential:
        """Constructs the Siren neural network.

        Args:
            n_in (int): Number of input features.
            n_hidden (list[int]): Number of neurons in each layer.
            n_out (int): Number of output features.
            first_omega_0 (float, optional): Scaling factor of the Sine function of the first layer. Defaults to 30.
            hidden_omega_0 (float, optional): Scaling factor of the Sine function of the hidden layers. Defaults to 30.
        Returns:
            torch.Sequential: Pytorch module
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
