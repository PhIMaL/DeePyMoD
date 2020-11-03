""" Contains the library classes that store the parameters u_t, theta"""
import numpy as np
import torch
from torch.autograd import grad
from itertools import combinations
from functools import reduce
from .deepmod import Library
from typing import Tuple
from ..utils.types import TensorList


# ==================== Library helper functions =======================
def library_poly(prediction: torch.Tensor, max_order: int) -> torch.Tensor:
    """ Compute the library up to some polynomial order

    Args:
        prediction (torch.Tensor): the data u for which to evaluate the library
        max_order (int): the maximum polynomial order up to which compute the library

    Returns:
        torch.Tensor: [description]
    """
    u = torch.ones_like(prediction)
    for order in np.arange(1, max_order+1):
        u = torch.cat((u, u[:, order-1:order] * prediction), dim=1)

    return u


def library_deriv(data: torch.Tensor, prediction: torch.Tensor, max_order: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """[summary]

    Args:
        data (torch.Tensor): [description]
        prediction (torch.Tensor): [description]
        max_order (int): [description]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: [description]
    """
    dy = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
    time_deriv = dy[:, 0:1]

    if max_order == 0:
        du = torch.ones_like(time_deriv)
    else:
        du = torch.cat((torch.ones_like(time_deriv), dy[:, 1:2]), dim=1)
        if max_order > 1:
            for order in np.arange(1, max_order):
                du = torch.cat((du, grad(du[:, order:order+1], data,
                                grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 1:2]), dim=1)

    return time_deriv, du


# ========================= Actual library functions ========================
class Library1D(Library):
    """ Construct a 1-dimensional library.

    Args:
        Library ([type]): Library object that can compute the library
    """
    def __init__(self, poly_order: int, diff_order: int) -> None:
        """ Create a library up with polynomial order containing up to 
        differentials order. i.e. for poly_order=1, diff_order=3       
        [$1, u_x, u_{xx}, u_{xxx}, u, u u_{x}, u u_{xx}, u u_{xxx}$]
        Args:
            poly_order (int): maximum order of the polynomial in the library
            diff_order (int): maximum order of the differentials in the library
            """
        super().__init__()
        self.poly_order = poly_order
        self.diff_order = diff_order

    def library(self, input: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[TensorList, TensorList]:
        """ Compute the library for the given a prediction and data

        Args:
            input (Tuple[torch.Tensor, torch.Tensor]): A prediction and its data

        Returns:
            Tuple[TensorList, TensorList]: The time derivatives and the thetas
            computed from the library and data.
        """
        prediction, data = input
        poly_list = []
        deriv_list = []
        time_deriv_list = []

        # Creating lists for all outputs
        for output in np.arange(prediction.shape[1]):
            time_deriv, du = library_deriv(data, prediction[:, output:output+1], self.diff_order)
            u = library_poly(prediction[:, output:output+1], self.poly_order)

            poly_list.append(u)
            deriv_list.append(du)
            time_deriv_list.append(time_deriv)

        samples = time_deriv_list[0].shape[0]
        total_terms = poly_list[0].shape[1] * deriv_list[0].shape[1]

        # Calculating theta
        if len(poly_list) == 1:
            # If we have a single output, we simply calculate and flatten matrix product
            # between polynomials and derivatives to get library
            theta = torch.matmul(poly_list[0][:, :, None], deriv_list[0][:, None, :]).view(samples, total_terms)
        else:
            theta_uv = reduce((lambda x, y: (x[:, :, None] @ y[:, None, :]).view(samples, -1)), poly_list)
            # calculate all unique combinations of derivatives
            theta_dudv = torch.cat([torch.matmul(du[:, :, None], dv[:, None, :]).view(samples, -1)[:, 1:]
                                    for du, dv in combinations(deriv_list, 2)], 1)
            theta = torch.cat([theta_uv, theta_dudv], dim=1)

        return time_deriv_list, [theta]


class Library2D(Library):
    """ Construct a 2-dimensional library.

    Args:
        Library ([type]): Library object that can compute the library
    """
    def __init__(self, poly_order: int) -> None:
        """ Create a 2D library up with polynomial order containing up to 
        differentials order. i.e. for poly_order=1     
        [$1, u_x, u_y, u_{xx}, u_{yy}, u_{xy}$]
        Args:
            poly_order (int): maximum order of the polynomial in the library
            """
        super().__init__()
        self.poly_order = poly_order

    def library(self, input: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[TensorList, TensorList]:
        """ Compute the library for the given a prediction and data

        Args:
            input (Tuple[torch.Tensor, torch.Tensor]): A prediction and its data

        Returns:
            Tuple[TensorList, TensorList]: The time derivatives and the thetas
            computed from the library and data.
        """

        prediction, data = input
        # Polynomial

        u = torch.ones_like(prediction)
        for order in np.arange(1, self.poly_order+1):
            u = torch.cat((u, u[:, order-1:order] * prediction), dim=1)

        # Gradients
        du = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
        u_t = du[:, 0:1]
        u_x = du[:, 1:2]
        u_y = du[:, 2:3]
        du2 = grad(u_x, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
        u_xx = du2[:, 1:2]
        u_xy = du2[:, 2:3]
        u_yy = grad(u_y, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 2:3]

        du = torch.cat((torch.ones_like(u_x), u_x, u_y, u_xx, u_yy, u_xy), dim=1)

        samples = du.shape[0]
        # Bringing it together
        theta = torch.matmul(u[:, :, None], du[:, None, :]).view(samples, -1)

        return [u_t], [theta]
