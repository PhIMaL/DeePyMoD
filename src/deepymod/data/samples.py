import torch
import numpy as np
from numpy import ndarray
from numpy.random import default_rng
from abc import ABC, ABCMeta
from deepymod.data.base import Subsampler, Dataset


class Subsample_grid(Subsampler):
    @staticmethod
    def sample(grid, grid_data, number_of_samples):
        print(number_of_samples)
        """Subsample on the second axis for data in the format [u, x, t]"""
        # getting indices of samples
        x_idx = torch.linspace(
            0, grid.shape[1] - 1, number_of_samples, dtype=torch.long
        )  # getting x locations
        # getting sample locations from indices
        subsampled_coords = torch.tensor(grid[:, :, x_idx].reshape(-1, 2))
        subsampled_data = torch.tensor(grid_data[:, :, x_idx].reshape(-1, 1))
        return subsampled_coords, subsampled_data


class Subsample_shifted_grid(Subsampler):
    @staticmethod
    def sub_sample_shifted_grid(grid, grid_data, number_of_samples):
        # getting indices of samples
        x_idx = torch.linspace(
            0, grid.shape[1] - 1, number_of_samples, dtype=torch.long
        )  # getting x locations
        # getting sample locations from indices
        subsampled_coords = torch.tensor(grid[:, :, x_idx].reshape(-1, 2))
        subsampled_data = torch.tensor(grid_data[:, :, x_idx].reshape(-1, 1))
        return subsampled_coords, subsampled_data


class Subsample_random(Subsampler):
    @staticmethod
    def sub_sample_random(grid, grid_data, number_of_samples):
        # getting indices of samples
        x_idx = torch.linspace(
            0, grid.shape[1] - 1, number_of_samples, dtype=torch.long
        )  # getting x locations
        # getting sample locations from indices
        subsampled_coords = torch.tensor(grid[:, :, x_idx].reshape(-1, 2))
        subsampled_data = torch.tensor(grid_data[:, :, x_idx].reshape(-1, 1))
        return subsampled_coords, subsampled_data


class MatlabDataset2D(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self):
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
        coords = torch.tensor(np.stack((X0, X1)))  # .reshape(-1, 2))
        data = torch.tensor(y).unsqueeze(0)  # .reshape(-1, 1))
        return coords, data
