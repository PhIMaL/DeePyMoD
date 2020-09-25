""" This file contains the building blocks for the deepmod framework. These are all abstract
    classes and implement the flow logic, rather than the specifics.
"""

import torch.nn as nn
import torch
from typing import Tuple
from ..utils.types import TensorList
from abc import ABCMeta, abstractmethod
import numpy as np


class Constraint(nn.Module, metaclass=ABCMeta):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self) -> None:
        super().__init__()
        self.sparsity_masks: TensorList = None

    def forward(self, input: Tuple[TensorList, TensorList]) -> Tuple[TensorList, TensorList]:
        """[summary]

        Args:
            input (Tuple[TensorList, TensorList]): [description]

        Returns:
            Tuple[TensorList, TensorList]: [description]
        """
        time_derivs, thetas = input

        if self.sparsity_masks is None:
            self.sparsity_masks = [torch.ones(theta.shape[1], dtype=torch.bool) for theta in thetas]

        sparse_thetas = self.apply_mask(thetas)
        self.coeff_vectors = self.calculate_coeffs(sparse_thetas, time_derivs)
        return sparse_thetas, self.coeff_vectors

    def apply_mask(self, thetas: TensorList) -> TensorList:
        """[summary]

        Args:
            thetas (TensorList): [description]

        Returns:
            TensorList: [description]
        """
        sparse_thetas = [theta[:, sparsity_mask] for theta, sparsity_mask in zip(thetas, self.sparsity_masks)]
        return sparse_thetas

    @abstractmethod
    def calculate_coeffs(self, sparse_thetas: TensorList, time_derivs: TensorList) -> TensorList: pass


class Estimator(nn.Module,  metaclass=ABCMeta):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, thetas: TensorList, time_derivs: TensorList) -> TensorList:
        """[summary]

        Args:
            thetas (TensorList): [description]
            time_derivs (TensorList): [description]

        Returns:
            TensorList: [description]
        """

        # we first normalize theta and the time deriv
        with torch.no_grad():
            normed_time_derivs = [(time_deriv / torch.norm(time_deriv)).detach().cpu() for time_deriv in time_derivs]
            normed_thetas = [(theta / torch.norm(theta, dim=0, keepdim=True)).detach().cpu() for theta in thetas]
        
        self.coeff_vectors = [self.fit(theta, time_deriv.squeeze())
                              for theta, time_deriv in zip(normed_thetas, normed_time_derivs)]
        sparsity_masks = [torch.tensor(coeff_vector != 0.0, dtype=torch.bool)
                          for coeff_vector in self.coeff_vectors]

        return sparsity_masks

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray: pass


class Library(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tuple[TensorList, TensorList]) -> Tuple[TensorList, TensorList]:
        """[summary]

        Args:
            input (torch.Tensor): [description]

        Returns:
            Tuple[TensorList, TensorList]: [description]
        """
        time_derivs, thetas = self.library(input)
        return time_derivs, thetas

    @abstractmethod
    def library(self, input: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[TensorList, TensorList]: pass


class DeepMoD(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self,
                 function_approximator: torch.nn.Sequential,
                 library: Library,
                 sparsity_estimator: Estimator,
                 constraint: Constraint) -> None:
        super().__init__()
        self.func_approx = function_approximator
        self.library = library
        self.sparse_estimator = sparsity_estimator
        self.constraint = constraint

    def forward(self, input: torch.Tensor) -> Tuple[TensorList, TensorList, TensorList, TensorList, TensorList]:
        """[summary]

        Args:
            input (torch.Tensor): [description]

        Returns:
            Tuple[TensorList, TensorList, TensorList, TensorList, TensorList]: [description]
        """
        prediction = self.func_approx(input)
        time_derivs, thetas = self.library((prediction, input))
        sparse_thetas, constraint_coeffs = self.constraint((time_derivs, thetas))
        return prediction, time_derivs, sparse_thetas, thetas, constraint_coeffs
