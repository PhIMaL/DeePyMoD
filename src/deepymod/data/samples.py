import torch
import numpy as np
from numpy import ndarray
from numpy.random import default_rng
from abc import ABC, ABCMeta, abstractmethod


class Subsampler(ABC, metaclass=ABCMeta):
    @abstractmethod
    def sample():
        raise NotImplementedError


class Subsample_axis(Subsampler):
    @staticmethod
    def sample(coords, data, number_of_slices):
        print(number_of_slices)
        """Subsample on the second axis for that has shape [t,x,y,feature] for both data and features.
        it will format """
        # getting indices of samples
        x_idx = torch.linspace(
            0, coords.shape[0] - 1, number_of_slices, dtype=torch.long
        )  # getting x locations
        # getting sample locations from indices
        return coords[x_idx], data[x_idx]


class Subsample_shifted_grid(Subsampler):
    @staticmethod
    def sample(coords, data, number_of_samples):
        return NotImplementedError


class Subsample_random(Subsampler):
    @staticmethod
    def sample(coords, data, number_of_samples):
        # Ensure that both are of the shape (number_of_samples, number_of_features) before random sampling.
        if len(data.shape) != 2 or len(coords.shape) != 2:
            coords = coords.reshape((-1, coords.shape[-1]))
            data = data.reshape((-1, data.shape[-1]))
        # getting indices of samples
        x_idx = torch.randperm(coords.shape[0])[
            :number_of_samples
        ]  # getting x locations
        # getting sample locations from indices
        subsampled_coords = coords[x_idx]
        subsampled_data = data[x_idx]
        return subsampled_coords, subsampled_data
