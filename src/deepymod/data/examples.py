import torch
import numpy as np
from deepymod.data.base import Dataset


def sinc_2d():
    """Output: Grid[N x M x L],  Data[N x M x O],
    N = Coordinate dimension 0
    M = Coordinate dimension 1
    L = Input data dimension
    O = Output data dimension
    """
    x0 = np.linspace(0, 2 * np.pi, 100)
    x1 = np.linspace(-np.pi, np.pi, 100)
    X0, X1 = np.meshgrid(x0, x1)
    y = np.sinc(X0 * X1)
    coords = torch.tensor(np.stack((X0, X1)))
    data = torch.tensor(y).unsqueeze(0)
    return coords, data
