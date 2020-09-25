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
            self.sparsity_masks = [torch.ones(theta.shape[1], dtype=torch.bool).to(theta.device) for theta in thetas]

        sparse_thetas = self.apply_mask(thetas)
        self.coeff_vectors = self.calculate_coeffs(sparse_thetas, time_derivs)


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
        self.coeff_vectors = None

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
            normed_time_derivs = [(time_deriv / torch.norm(time_deriv)).detach().cpu().numpy() for time_deriv in time_derivs]
            normed_thetas = [(theta / torch.norm(theta, dim=0, keepdim=True)).detach().cpu().numpy() for theta in thetas]
        
        self.coeff_vectors = [self.fit(theta, time_deriv.squeeze())[:, None]
                              for theta, time_deriv in zip(normed_thetas, normed_time_derivs)]
        sparsity_masks = [torch.tensor(coeff_vector != 0.0, dtype=torch.bool).squeeze().to(thetas[0].device) # move to gpu if required
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
        self.norms = None

    def forward(self, input: Tuple[TensorList, TensorList]) -> Tuple[TensorList, TensorList]:
        """[summary]

        Args:
            input (torch.Tensor): [description]

        Returns:
            Tuple[TensorList, TensorList]: [description]
        """
        time_derivs, thetas = self.library(input)
        self.norms = [(torch.norm(time_deriv) / torch.norm(theta, dim=0, keepdim=True)).detach().squeeze() for time_deriv, theta in zip(time_derivs, thetas)]
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

    def forward(self, input: torch.Tensor) -> Tuple[TensorList, TensorList, TensorList]:
        """[summary]

        Args:
            input (torch.Tensor): [description]

        Returns:
            Tuple[TensorList, TensorList, TensorList, TensorList, TensorList]: [description]
        """
        prediction, coordinates = self.func_approx(input)
        time_derivs, thetas = self.library((prediction, coordinates))
        self.constraint((time_derivs, thetas))
        return prediction, time_derivs, thetas
    
    @property
    def sparsity_masks(self):
        return self.constraint.sparsity_masks
    
    def estimator_coeffs(self):
        coeff_vectors = self.sparse_estimator.coeff_vectors
        return coeff_vectors

    def constraint_coeffs(self, scaled=False, sparse=False):
        coeff_vectors = self.constraint.coeff_vectors
        if scaled:
            coeff_vectors = [coeff / norm[:, None] for coeff, norm, mask in zip(coeff_vectors, self.library.norms, self.sparsity_masks)]
        if sparse:
            coeff_vectors = [sparsity_mask[:, None] * coeff for sparsity_mask, coeff in zip(self.sparsity_masks, coeff_vectors)]
            #coeff_vectors = [torch.zeros((mask.shape[0], 1)).to(coeff_vector.device).masked_scatter_(mask[:, None], coeff_vector)
            #                 for mask, coeff_vector
            #                 in zip(self.sparsity_masks, coeff_vectors)]
        return coeff_vectors


