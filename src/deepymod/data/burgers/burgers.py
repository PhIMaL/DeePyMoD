""" Contains several interactive datasets for the Burgers equation including:
    - Burgers with initial delta peak profile
    - Burgers with initial cosine profile
    - Burgers with initial sawtooth profile"""

import torch
from numpy import pi


def BurgersDelta(x: torch.tensor, t: torch.tensor, v: float, A: float) -> torch.tensor:
    """ Function to generate analytical solutions of Burgers equation with delta peak initial condition: u(x, 0) = A delta(x)

    Source: https://www.iist.ac.in/sites/default/files/people/IN08026/Burgers_equation_viscous.pdf
    Note that this source has an error in the erfc prefactor, should be sqrt(pi)/2, not sqrt(pi/2).

    Args:
        x ([Tensor]): Input vector of spatial coordinates.
        t ([Tensor]): Input vector of temporal coordinates.
        v (Float): Velocity.
        A (Float): Amplitude of the initial condition.

    Returns:
        [Tensor]: solution. 
    """
    R = torch.tensor(A / (2 * v))  # otherwise throws error
    z = x / torch.sqrt(4 * v * t)

    u = torch.sqrt(v / (pi * t)) * ((torch.exp(R) - 1) * torch.exp(-z**2)) / (1 + (torch.exp(R) - 1) / 2 * torch.erfc(z))
    return u


def BurgersCos(x: torch.tensor, t: torch.tensor, v: float, a: float, b: float, k: float):
    """Function to generate analytical solutions of Burgers equation with cosine initial condition: 
    $u(x, 0) = b + a \cos(kx)$

    Source: https://www.iist.ac.in/sites/default/files/people/IN08026/Burgers_equation_viscous.pdf

    Args:
        x ([Tensor]): Input vector of spatial coordinates.
        t ([Tensor]): Input vector of temporal coordinates.
        v (Float): Velocity. 
        a ([Float]): Amplitude of the initial periodic condition.
        b ([Float]): Offset of the initial condition.
        k ([Float]): Wavenumber of the initial condition.

    Returns:
        [Tensor]: solution. 
    """

    z = v * k**2 * t

    u = (2 * v * a * k * torch.exp(-z) * torch.sin(k*x))/(b + a * torch.exp(-z) * torch.cos(k * x))
    return u


def BurgersSawtooth(x: torch.tensor, t: torch.tensor, v: float) -> torch.tensor:
    """Function to generate analytical solutions of Burgers equation with sawtooth initial condition (see soruce for exact expression). Solution only
    valid between for x in [0, 2pi] and t in [0, 0.5]

    http://www.thevisualroom.com/02_barba_projects/burgers_equation.html

    Args:
        x ([Tensor]): Input vector of spatial coordinates.
        t ([Tensor]): Input vector of temporal coordinates.
        v (Float): Velocity. 

    Returns:
        [Tensor]: solution. 
    """


    z_left = (x - 4 * t)
    z_right = (x - 4 * t - 2 * pi)
    l = 4 * v * (t + 1)

    phi = torch.exp(-z_left**2/l) + torch.exp(-z_right**2/l)
    dphi_x = - 2 * z_left / l * torch.exp(-z_left**2 / l) - 2 * z_right / l * torch.exp(-z_right**2 / l)
    u = -2 * v * dphi_x / phi + 4
    return u
