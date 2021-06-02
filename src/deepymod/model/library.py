""" Contains the library classes that store the parameters u_t, theta"""
import numpy as np
import torch
from torch import autograd
from itertools import combinations
from functools import reduce
from .deepmod import Library
from typing import Tuple
from ..utils.types import TensorList


# ==================== Library helper functions =======================
def library_poly(prediction: torch.Tensor, max_order: int) -> torch.Tensor:
    """Given a prediction u, returns u^n up to max_order, including ones as first column.

    Args:
        prediction (torch.Tensor): the data u for which to evaluate the library (n_samples x 1)
        max_order (int): the maximum polynomial order up to which compute the library

    Returns:
        torch.Tensor: Tensor with polynomials (n_samples, max_order + 1)
    """

    polynomials = [prediction ** order for order in torch.arange(1, max_order + 1)]
    u = torch.cat([torch.ones_like(prediction)] + polynomials, dim=1)
    return u


def derivs(
    prediction: torch.Tensor, coordinates: torch.Tensor, max_order: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given a prediction u evaluated at coordinates (t, x), returns du/dt and du/dx up to max_order, including ones
    as first column.

    Args:
        data (torch.Tensor): (t, x) locations of where to evaluate derivatives (n_samples x 2)
        prediction (torch.Tensor): the data u for which to evaluate the library (n_samples x 1)
        max_order (int): maximum order of derivatives to be calculated.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: time derivative and feature library ((n_samples, 1), (n_samples,  max_order + 1))
    """
    assert max_order > 0, "Only 1st order and up allowed."

    grad = lambda f: autograd.grad(
        f, coordinates, grad_outputs=torch.ones_like(f), create_graph=True
    )[0]

    df = grad(prediction)
    time_derivs, dx = df[:, [0]], df[:, [1]]

    du = [torch.ones_like(prediction), dx]
    for order in np.arange(1, max_order):
        du.append(grad(du[order])[:, [1]])
    space_derivs = torch.cat(du, dim=1)

    return time_derivs, space_derivs


# ========================= Actual library functions ========================
class Library1D(Library):
    def __init__(self, poly_order: int, diff_order: int) -> None:
        """Calculates the temporal derivative a library/feature matrix consisting of
        1) polynomials up to order poly_order, i.e. u, u^2...
        2) derivatives up to order diff_order, i.e. u_x, u_xx
        3) cross terms of 1) and 2), i.e. $uu_x$, $u^2u_xx$

        Order of terms is derivative first, i.e. [$1, u_x, u, uu_x, u^2, ...$]

        Only works for 1D+1 data. Also works for multiple outputs but in that case doesn't calculate
        polynomial and derivative cross terms.

        Args:
            poly_order (int): maximum order of the polynomial in the library
            diff_order (int): maximum order of the differentials in the library
        """

        super().__init__()
        self.poly_order = poly_order
        self.diff_order = diff_order

    def library(
        self, input: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[TensorList, TensorList]:
        """Compute the temporal derivative and library for the given prediction at locations given by data.
            Data should have t in first column, x in second.

        Args:
            input (Tuple[torch.Tensor, torch.Tensor]): A prediction u (n_samples, n_outputs) and spatiotemporal locations (n_samples, 2).

        Returns:
            Tuple[TensorList, TensorList]: The time derivatives [(n_samples, 1) x n_outputs] and the thetas [(n_samples, (poly_order + 1)(deriv_order + 1))]
            computed from the library and data.
        """
        prediction, coordinates = input
        time_derivs, space_derivs = self.derivative_features(
            prediction, coordinates, self.diff_order
        )
        thetas = self.build_features(prediction, space_derivs, self.poly_order)

        return time_derivs, thetas

    @staticmethod
    def derivative_features(
        prediction: torch.Tensor, coordinates: torch.Tensor, diff_order: int
    ) -> Tuple[TensorList, TensorList]:

        # Calculate derivs over all outputs
        n_outputs = prediction.shape[1]
        df = [
            derivs(prediction[:, [output]], coordinates, diff_order)
            for output in np.arange(n_outputs)
        ]
        # Unzip to separate time and space
        time_derivs, space_derivs = map(list, zip(*df))
        return time_derivs, space_derivs

    @staticmethod
    def build_features(
        prediction: torch.Tensor, space_derivs: torch.Tensor, poly_order: int
    ) -> TensorList:

        n_samples, n_outputs = prediction.shape
        total_terms = (poly_order + 1) * space_derivs[0].shape[1]

        # Creating lists for all outputs
        poly_list = [
            library_poly(prediction[:, [output]], poly_order)
            for output in np.arange(n_outputs)
        ]

        # Calculating theta
        if n_outputs == 1:
            # If we have a single output, we simply calculate and flatten matrix product
            # between polynomials and derivatives to get library
            theta = torch.matmul(
                poly_list[0][:, :, None], space_derivs[0][:, None, :]
            ).view(n_samples, total_terms)

        else:
            uv = reduce(
                (lambda x, y: (x[:, :, None] @ y[:, None, :]).view(n_samples, -1)),
                poly_list,
            )
            # calculate all unique combinations of derivatives
            dudv = [
                torch.matmul(du[:, :, None], dv[:, None, :]).view(n_samples, -1)[:, 1:]
                for du, dv in combinations(space_derivs, 2)
            ]
            dudv = torch.cat(dudv, dim=1)
            theta = torch.cat([uv, dudv], dim=1)
        return [theta]


class Library2D(Library):
    def __init__(self, poly_order: int) -> None:
        """Create a 2D library up to given polynomial order with second order derivatives
         i.e. for poly_order=1: [$1, u_x, u_y, u_{xx}, u_{yy}, u_{xy}$]
        Args:
            poly_order (int): maximum order of the polynomial in the library
        """
        super().__init__()
        self.poly_order = poly_order

    def library(
        self, input: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[TensorList, TensorList]:
        """Compute the library for the given a prediction and data

        Args:
            input (Tuple[torch.Tensor, torch.Tensor]): A prediction and its data

        Returns:
            Tuple[TensorList, TensorList]: The time derivatives and the thetas
            computed from the library and data.
        """

        prediction, data = input
        # Polynomial

        u = torch.ones_like(prediction)
        for order in np.arange(1, self.poly_order + 1):
            u = torch.cat((u, u[:, order - 1 : order] * prediction), dim=1)

        # Gradients
        du = autograd.grad(
            prediction,
            data,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
        )[0]
        u_t = du[:, 0:1]
        u_x = du[:, 1:2]
        u_y = du[:, 2:3]
        du2 = autograd.grad(
            u_x, data, grad_outputs=torch.ones_like(prediction), create_graph=True
        )[0]
        u_xx = du2[:, 1:2]
        u_xy = du2[:, 2:3]
        u_yy = autograd.grad(
            u_y, data, grad_outputs=torch.ones_like(prediction), create_graph=True
        )[0][:, 2:3]

        du = torch.cat((torch.ones_like(u_x), u_x, u_y, u_xx, u_yy, u_xy), dim=1)

        samples = du.shape[0]
        # Bringing it together
        theta = torch.matmul(u[:, :, None], du[:, None, :]).view(samples, -1)

        return [u_t], [theta]
