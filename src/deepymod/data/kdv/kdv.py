""" Contains several interactive datasets for the Korteweg-de-Vries equation including:
    - A single soliton wave
    - Two soliton waves
"""

import torch
import numpy as np


def SingleSoliton(x: torch.tensor, t: torch.tensor, c: float, x0: float) -> torch.tensor:
    """Single soliton solution of the KdV equation (u_t + u_{xxx} - 6 u u_x = 0)

    Args:
        x ([Tensor]): Input vector of spatial coordinates.
        t ([Tensor]): Input vector of temporal coordinates.
        c ([Float]): Velocity. 
        x0 ([Float]): Offset.
    Returns:
        [Tensor]: Solution. 
    """
    xi = np.sqrt(c) / 2 * (x - c * t - x0)  # switch to moving coordinate frame
    u = c / 2 * 1 / torch.cosh(xi)**2
    return u


def DoubleSoliton(x: torch.tensor, t: torch.tensor, c: float, x0: float) -> torch.tensor:
    """ Single soliton solution of the KdV equation (u_t + u_{xxx} - 6 u u_x = 0)
    source: http://lie.math.brocku.ca/~sanco/solitons/kdv_solitons.php

    Args:
        x ([Tensor]): Input vector of spatial coordinates.
        t ([Tensor]): Input vector of temporal coordinates.
        c ([Array]): Array containing the velocities of the two solitons, note that c[0] > c[1]. 
        x0 ([Array]):  Array containing the offsets of the two solitons.

    Returns:
        [Tensor]: Solution.
    """
    assert c[0] > c[1], 'c1 has to be bigger than c[2]'
    
    xi0 = np.sqrt(c[0]) / 2 * (x - c[0] * t - x0[0]) #  switch to moving coordinate frame
    xi1 = np.sqrt(c[1]) / 2 * (x - c[1] * t - x0[1])

    part_1 = 2 * (c[0] - c[1])
    numerator = c[0] * torch.cosh(xi1)**2 + c[1] * torch.sinh(xi0)**2
    denominator_1 = (np.sqrt(c[0]) - np.sqrt(c[1])) * torch.cosh(xi0 + xi1)
    denominator_2 = (np.sqrt(c[0]) + np.sqrt(c[1])) * torch.cosh(xi0 - xi1)
    u = part_1 * numerator / (denominator_1 + denominator_2)**2
    return u