import torch
import numpy as np
from numpy import ndarray
from numpy.random import default_rng
from abc import ABC, ABCMeta, abstractmethod


class Subsampler(ABC, metaclass=ABCMeta):
    @abstractmethod
    def sample():
        raise NotImplementedError


class Subsample_grid(Subsampler):
    @staticmethod
    def sample(coords, data, number_of_samples):
        print(number_of_samples)
        """Subsample on the second axis for data in the format [u, x, t]"""
        # getting indices of samples
        x_idx = torch.linspace(
            0, coords.shape[1] - 1, number_of_samples, dtype=torch.long
        )  # getting x locations
        # getting sample locations from indices
        subsampled_coords = coords[:, :, x_idx]
        subsampled_data = data[:, :, x_idx]
        return subsampled_coords, subsampled_data


class Subsample_shifted_grid(Subsampler):
    @staticmethod
    def sub_sample_shifted_grid(coords, data, number_of_samples):
        return NotImplementedError


class Subsample_random(Subsampler):
    @staticmethod
    def sub_sample_random(coords, data, number_of_samples):
        # getting indices of samples
        x_idx = torch.randperm(coords.shape[1])[
            :number_of_samples
        ]  # getting x locations
        # getting sample locations from indices
        subsampled_coords = coords[:, :, x_idx]
        subsampled_data = data[:, :, x_idx]
        return subsampled_coords, subsampled_data
