""" Contains several interactive datasets for the Diffusion equation including:
    - Diffusion
    - Advection-Diffusion in 2 dimensions
"""

from numpy import pi
import torch


def DiffusionGaussian(x: torch.tensor, t: torch.tensor, D: float, x0: float, sigma: float) -> torch.tensor:
    """Function to generate the solution to the 1D diffusion equation.

    REFERENCE

    Args:
        x ([Tensor]): Input vector of spatial coordinates.
        t ([Tensor]): Input vector of temporal coordinates.
        D (Float): Diffusion coefficient 
        x0 (Float): Spatial coordinate where the gaussian is centered
        sigma (Float): Scale parameter that adjusts the initial shape of the parameter

    Returns:
        [Tensor]: solution. 
    """
    u = (2 * pi * sigma**2 + 4 * pi * D * t)**(-1/2) * torch.exp(-(x - x0)**2/(2 * sigma**2 + 4 * D * t))
    return u

def AdvectionDiffusionGaussian2D(x: torch.tensor, t: torch.tensor, D: float, x0: torch.tensor, sigma: float, v: torch.tensor) -> torch.tensor:
    """Function to generate the solution to the 2D diffusion equation.

    REFERENCE

    Args:
        x ([Tensor]): [N, 2] Input vector of spatial coordinates.
        t ([Tensor]): Input vector of temporal coordinates.
        D (Float): Diffusion coefficient 
        x0 ([Tensor]): Spatial coordinate where the gaussian is centered
        sigma (Float): Scale parameter that adjusts the initial shape of the parameter
        v ([Tensor]): [2] Initial velocity of the gaussian.

    Returns:
        [Tensor]: solution. 
    """
    u = (2 * pi * sigma**2 + 4 * pi * D * t)**(-1) * torch.exp(-((x[:, 0:1] - x0[0] - v[0] * t)**2 + (x[:, 1:2] - x0[1] - v[1] * t)**2)/(2 * sigma**2 + 4 * D * t))
    return u
