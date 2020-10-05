from numpy import pi
import torch


def DiffusionGaussian(x, t, D, x0, sigma):
    u = (2 * pi * sigma**2 + 4 * pi * D * t)**(-1/2) * torch.exp(-(x - x0)**2/(2 * sigma**2 + 4 * D * t))
    return u

def AdvectionDiffusionGaussian2D(x, t, D, x0, sigma, v):
    u = (2 * pi * sigma**2 + 4 * pi * D * t)**(-1) * torch.exp(-((x[:, 0:1] - x0[0] - v[0] * t)**2 + (x[:, 1:2] - x0[1] - v[1] * t)**2)/(2 * sigma**2 + 4 * D * t))
    return u
