import torch
import numpy as np


def SingleSoliton(x, t, c, x0):
    """[summary]
    source:
    Args:
        x ([type]): [description]
        t ([type]): [description]
        c ([type]): [description]
        x0 ([type]): [description]
    """
    xi = np.sqrt(c) / 2 * (x - c * t - x0) # switch to moving coordinate frame

    u = c / 2 * 1 / torch.cosh(xi)**2
    return u


def DoubleSoliton(x, t, c, x0):
    """[summary]
    source: http://lie.math.brocku.ca/~sanco/solitons/kdv_solitons.php
    Args:
        x ([type]): [description]
        t ([type]): [description]
        c ([type]): [description]
        x0 ([type]): [description]
    """
    assert c[0] > c[1], 'c1 has to be bigger than c[2]'
    
    xi0 = np.sqrt(c[0]) / 2 * (x - c[0] * t - x0[0]) # switch to moving coordinate frame
    xi1 = np.sqrt(c[1]) / 2 * (x - c[1] * t - x0[1])

    part_1 = 2 * (c[0] - c[1])
    numerator = c[0] * torch.cosh(xi1)**2 + c[1] * torch.sinh(xi0)**2
    denominator_1 = (np.sqrt(c[0]) - np.sqrt(c[1])) * torch.cosh(xi0 + xi1)
    denominator_2 = (np.sqrt(c[0]) + np.sqrt(c[1])) * torch.cosh(xi0 - xi1)
    u = part_1 * numerator / (denominator_1 + denominator_2)**2
    return u