"""This module contains concrete implementations of the constraint component. 
"""


import torch
from .deepmod import Constraint
from typing import List
TensorList = List[torch.Tensor]


class LeastSquares(Constraint):
    """Implements the constraint as a least squares problem solved by QR decomposition. """

    def __init__(self) -> None:
        super().__init__()

    def calculate_coeffs(self, sparse_thetas: TensorList, time_derivs: TensorList) -> TensorList:
        """Calculates the coefficients of the constraint using the QR decomposition for every pair
        of sparse feature matrix and time derivative.

        Args:
            sparse_thetas (TensorList): List containing the sparse feature tensors. 
            time_derivs (TensorList): List containing the time derivatives.

        Returns:
            [TensorList]: Calculated coefficients.
        """
        opt_coeff = []
        for theta, dt in zip(sparse_thetas, time_derivs):
            Q, R = torch.qr(theta)  # solution of lst. sq. by QR decomp.
            opt_coeff.append(torch.inverse(R) @ Q.T @ dt)

        # Putting them in the right spot
        coeff_vectors = [torch.zeros((mask.shape[0], 1)).to(coeff_vector.device).masked_scatter_(mask[:, None], coeff_vector)
                             for mask, coeff_vector
                             in zip(self.sparsity_masks, opt_coeff)]
        return coeff_vectors

class GradParams(Constraint):
    def __init__(self, n_params, n_eqs) -> None:
        super().__init__()
        self.coeff_vectors = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(n_params, 1)) for _ in torch.arange(n_eqs)])

    def calculate_coeffs(self, sparse_thetas, time_derivs):
        return self.coeff_vectors