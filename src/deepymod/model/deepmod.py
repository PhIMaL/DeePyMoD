""" This file contains building blocks for the deepmod framework:
    I) The constraint class that constrains the neural network with the obtained solution,
    II) The sparsity estimator class,
    III) Function library class on which the model discovery is performed.
    IV) The DeepMoD class integrates these seperate building blocks.
    These are all abstract classes and implement the flow logic, rather than the specifics.
"""

import torch.nn as nn
import torch
from typing import Tuple
from ..utils.types import TensorList
from abc import ABCMeta, abstractmethod
import numpy as np


class Constraint(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        """Abstract baseclass for the constraint module."""
        super().__init__()
        self.sparsity_masks: TensorList = None

    def forward(self, input: Tuple[TensorList, TensorList]) -> Tuple[TensorList, TensorList]:
        """ The forward pass of the constraint module applies the sparsity mask to the feature matrix theta,
        and then calculates the coefficients according to the method in the child.

        Args:
            input (Tuple[TensorList, TensorList]): (time_derivs, library) tuple of size
                    ([(n_samples, 1) X n_outputs], [(n_samples, n_features) x n_outputs]).
        """

        time_derivs, thetas = input

        if self.sparsity_masks is None:
            self.sparsity_masks = [torch.ones(theta.shape[1], dtype=torch.bool).to(theta.device) for theta in thetas]

        sparse_thetas = self.apply_mask(thetas)
        self.coeff_vectors = self.calculate_coeffs(sparse_thetas, time_derivs)

    def apply_mask(self, thetas: TensorList) -> TensorList:
        """ Applies the sparsity mask to the feature (library) matrix.

        Args:
            thetas (TensorList): List of all library matrices of size [(n_samples, n_features) x n_outputs].

        Returns:
            TensorList: The sparse version of the library matrices of size [(n_samples, n_active_features) x n_outputs].
        """
        sparse_thetas = [theta[:, sparsity_mask] for theta, sparsity_mask in zip(thetas, self.sparsity_masks)]
        return sparse_thetas

    @abstractmethod
    def calculate_coeffs(self, sparse_thetas: TensorList, time_derivs: TensorList) -> TensorList:
        """Abstract method. Specific method should return the coefficients as calculated from the sparse feature
        matrices and temporal derivatives.

        Args:
            sparse_thetas (TensorList): List containing the sparse feature tensors of size (n_samples, n_active_features).
            time_derivs (TensorList): List containing the time derivatives of size (n_samples, n_outputs).

        Returns:
            (TensorList): Calculated coefficients of size (n_features, n_outputs).
        """
        pass


class Estimator(nn.Module,  metaclass=ABCMeta):
    def __init__(self) -> None:
        """Abstract baseclass for the sparse estimator module."""
        super().__init__()
        self.coeff_vectors = None

    def forward(self, thetas: TensorList, time_derivs: TensorList) -> TensorList:
        """The forward pass of the sparse estimator module first normalizes the library matrices
        and time derivatives by dividing each column (i.e. feature) by their l2 norm, than calculate the coefficient vectors
        according to the sparse estimation algorithm supplied by the child and finally returns the sparsity
        mask (i.e. which terms are active) based on these coefficients.

        Args:
            thetas (TensorList): List containing the sparse feature tensors of size  [(n_samples, n_active_features) x n_outputs].
            time_derivs (TensorList): List containing the time derivatives of size  [(n_samples, 1) x n_outputs].

        Returns:
            (TensorList): List containting the sparsity masks of a boolean type and size  [(n_samples, n_features) x n_outputs].
        """

        # we first normalize theta and the time deriv
        with torch.no_grad():
            normed_time_derivs = [(time_deriv / torch.norm(time_deriv)).detach().cpu().numpy() for time_deriv in time_derivs]
            normed_thetas = [(theta / torch.norm(theta, dim=0, keepdim=True)).detach().cpu().numpy() for theta in thetas]

        self.coeff_vectors = [self.fit(theta, time_deriv.squeeze())[:, None]
                              for theta, time_deriv in zip(normed_thetas, normed_time_derivs)]
        sparsity_masks = [torch.tensor(coeff_vector != 0.0, dtype=torch.bool).squeeze().to(thetas[0].device)  # move to gpu if required
                          for coeff_vector in self.coeff_vectors]

        return sparsity_masks

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Abstract method. Specific method should compute the coefficient based on feature matrix X and observations y.
        Note that we expect X and y to be numpy arrays, i.e. this module is non-differentiable.

        Args:
            x (np.ndarray): Feature matrix of size (n_samples, n_features)
            y (np.ndarray): observations of size (n_samples, n_outputs)

        Returns:
            (np.ndarray): Coefficients of size (n_samples, n_outputs)
        """
        pass


class Library(nn.Module):
    def __init__(self) -> None:
        """Abstract baseclass for the library module."""
        super().__init__()
        self.norms = None

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[TensorList, TensorList]:
        """Compute the library (time derivatives and thetas) from a given dataset. Also calculates the norms
        of these, later used to calculate the normalized coefficients.

        Args:
            input (Tuple[TensorList, TensorList]): (prediction, data) tuple of size ((n_samples, n_outputs), (n_samples, n_dims))

        Returns:
            Tuple[TensorList, TensorList]: Temporal derivative and libraries of size ([(n_samples, 1) x n_outputs]), [(n_samples, n_features)x n_outputs])
        """

        time_derivs, thetas = self.library(input)
        self.norms = [(torch.norm(time_deriv) / torch.norm(theta, dim=0, keepdim=True)).detach().squeeze() for time_deriv, theta in zip(time_derivs, thetas)]
        return time_derivs, thetas

    @abstractmethod
    def library(self, input: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[TensorList, TensorList]:
        """Abstract method. Specific method should calculate the temporal derivative and feature matrices.
        These should be a list; one temporal derivative and feature matrix per output.

        Args:
        input (Tuple[TensorList, TensorList]): (prediction, data) tuple of size ((n_samples, n_outputs), (n_samples, n_dims))

        Returns:
        Tuple[TensorList, TensorList]: Temporal derivative and libraries of size ([(n_samples, 1) x n_outputs]), [(n_samples, n_features)x n_outputs])
        """
        pass


class DeepMoD(nn.Module):
    def __init__(self,
                 function_approximator: torch.nn.Sequential,
                 library: Library,
                 sparsity_estimator: Estimator,
                 constraint: Constraint) -> None:
        """The DeepMoD class integrates the various buiding blocks into one module. The function approximator approximates the data,
        the library than builds a feature matrix from its output and the constraint constrains these. The sparsity estimator is called
        during training to update the sparsity mask (i.e. which terms the constraint is allowed to use.)

        Args:
            function_approximator (torch.nn.Sequential): [description]
            library (Library): [description]
            sparsity_estimator (Estimator): [description]
            constraint (Constraint): [description]
        """
        super().__init__()
        self.func_approx = function_approximator
        self.library = library
        self.sparse_estimator = sparsity_estimator
        self.constraint = constraint

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, TensorList, TensorList]:
        """ The forward pass approximates the data, builds the time derivative and feature matrices
        and applies the constraint.

        It returns the prediction of the network, the time derivatives and the feature matrices.

        Args:
            input (torch.Tensor):  Tensor of shape (n_samples, n_outputs) containing the coordinates, first column should be the time coordinate.

        Returns:
            Tuple[torch.Tensor, TensorList, TensorList]: The prediction, time derivatives and and feature matrices of respective sizes
                                                       ((n_samples, n_outputs), [(n_samples, 1) x n_outputs]), [(n_samples, n_features) x n_outputs])

        """
        prediction, coordinates = self.func_approx(input)
        time_derivs, thetas = self.library((prediction, coordinates))
        self.constraint((time_derivs, thetas))
        return prediction, time_derivs, thetas

    @property
    def sparsity_masks(self):
        """Returns the sparsity masks which contain the active terms. """
        return self.constraint.sparsity_masks

    def estimator_coeffs(self) -> TensorList:
        """ Calculate the coefficients as estimated by the sparse estimator.

        Returns:
            (TensorList): List of coefficients of size [(n_features, 1) x n_outputs]
        """
        coeff_vectors = self.sparse_estimator.coeff_vectors
        return coeff_vectors

    def constraint_coeffs(self, scaled=False, sparse=False) -> TensorList:
        """ Calculate the coefficients as estimated by the constraint.

        Args:
            scaled (bool): Determine whether or not the coefficients should be normalized
            sparse (bool): Whether to apply the sparsity mask to the coefficients.

        Returns:
            (TensorList): List of coefficients of size [(n_features, 1) x n_outputs]
        """
        coeff_vectors = self.constraint.coeff_vectors
        if scaled:
            coeff_vectors = [coeff / norm[:, None] for coeff, norm, mask in zip(coeff_vectors, self.library.norms, self.sparsity_masks)]
        if sparse:
            coeff_vectors = [sparsity_mask[:, None] * coeff for sparsity_mask, coeff in zip(self.sparsity_masks, coeff_vectors)]
        return coeff_vectors
