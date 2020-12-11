"""This module contains concrete implementations of the constraint component."""


import torch
from .deepmod import Constraint
from typing import List

TensorList = List[torch.Tensor]


class LeastSquares(Constraint):
    def __init__(self) -> None:
        """ Least Squares Constraint solved by QR decomposition"""
        super().__init__()

    def fit(self, sparse_thetas: TensorList, time_derivs: TensorList) -> TensorList:
        """Calculates the coefficients of the constraint using the QR decomposition for every pair
        of sparse feature matrix and time derivative.

        Args:
            sparse_thetas (TensorList): List containing the sparse feature tensors of size (n_samples, n_active_features).
            time_derivs (TensorList): List containing the time derivatives of size (n_samples, n_outputs).

        Returns:
            (TensorList): List of calculated coefficients of size [(n_active_features, 1) x n_outputs].
        """
        coeff_vectors = []
        for theta, dt in zip(sparse_thetas, time_derivs):
            Q, R = torch.qr(theta)  # solution of lst. sq. by QR decomp.
            coeff_vectors.append(torch.inverse(R) @ Q.T @ dt)
        return coeff_vectors


class GradParams(Constraint):
    def __init__(self, n_params: int, n_eqs: int) -> None:
        """Constrains the neural network by optimizing over the coefficients together with the network.
           Coefficient vectors are randomly initialized from a standard Gaussian.

        Args:
            n_params (int): number of features in feature matrix.
            n_eqs (int): number of outputs / equations to be discovered.
        """
        super().__init__()
        self.coeffs = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(n_params, 1)) for _ in torch.arange(n_eqs)]
        )

    def fit(self, sparse_thetas: TensorList, time_derivs: TensorList):
        """Returns the coefficients of the constraint, since we're optimizing them by
           gradient descent.

        Args:
            sparse_thetas (TensorList): List containing the sparse feature tensors of size (n_samples, n_active_features).
            time_derivs (TensorList): List containing the time derivatives of size (n_samples, n_outputs).

        Returns:
            (TensorList): Calculated coefficients of size (n_features, n_outputs).
        """
        return self.coeffs
