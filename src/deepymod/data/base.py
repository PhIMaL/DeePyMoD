""" Contains the base class for the Dataset (1 and 2 dimensional) and a function
     that takes a Pytorch tensor and converts it to a numpy array"""

import torch
import numpy as np
from numpy import ndarray
from numpy.random import default_rng
from deepymod.data.samples import Subsampler

from abc import ABC, ABCMeta, abstractmethod


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        load_function,
        apply_normalize=None,
        apply_noise=None,
        apply_shuffle=None,
        shuffle=True,
        subsampler: Subsampler = None,
        load_kwargs: dict = {},
        preprocess_kwargs: dict = {
            "random_state": 42,
            "noise_level": 0.0,
            "normalize_coords": False,
            "normalize_data": False,
        },
        subsampler_kwargs: dict = {},
        device: str = None,
    ):
        """A dataset class that loads the data, preprocesses it and lastly applies subsampling to it

        Args:
            load_function (func):Must return torch tensors in the format coordinates, data
            shuffle (bool, optional): Shuffle the data. Defaults to True.
            apply_normalize (func)
            subsampler (Subsampler, optional): Add some subsampling function. Defaults to None.
            load_kwargs (dict, optional): kwargs to pass to the load_function. Defaults to {}.
            preprocess_kwargs (dict, optional): (optional) arguments to pass to the preprocess method. Defaults to { "random_state": 42, "noise_level": 0.0, "normalize_coords": False, "normalize_data": False, }.
            subsampler_kwargs (dict, optional): (optional) arguments to pass to the subsampler method. Defaults to {}.
            device (str, optional): which device to send the data to. Defaults to None.
        """
        self.load = load_function
        self.subsampler = subsampler
        self.load_kwargs = load_kwargs
        self.preprocess_kwargs = preprocess_kwargs
        self.subsampler_kwargs = subsampler_kwargs  # so total number of samples is size(self.t_domain) * n_samples_per_frame
        # If some override function is provided, use that instead of the default.
        if apply_normalize != None:
            self.apply_normalize = apply_normalize
        if apply_noise != None:
            self.apply_normalize = apply_noise
        if apply_shuffle != None:
            self.apply_shuffle = apply_shuffle
        self.device = device
        self.shuffle = shuffle
        self.coords, self.data = self.load(**self.load_kwargs)
        # Ensure the data that loaded is not 0D/1D
        assert (
            len(self.coords.shape) >= 2
        ), "Please explicitely specify a feature axis for the coordinates"
        assert (
            len(self.data.shape) >= 2
        ), "Please explicitely specify a feature axis for the data"
        # Preprocess (add noise and normalization)
        self.coords, self.data = self.preprocess(
            self.coords, self.data, **self.preprocess_kwargs
        )
        # Apply the subsampler if there is one
        if self.subsampler:
            self.coords, self.data = self.subsampler.sample(
                self.coords, self.data, **self.subsampler_kwargs
            )
        # Reshaping the data to a (number_of_samples, number_of_features) shape if needed
        if len(self.data.shape) != 2 or len(self.coords.shape) != 2:
            self.coords, self.data = self._reshape(self.coords, self.data)
        if self.shuffle:
            self.coords, self.data = self.apply_shuffle(self.coords, self.data)
        # Now we know the data are shape (number_of_samples, number_of_features) we can set the number_of_samples
        self.number_of_samples = self.data.shape[0]

        print("Dataset is using device: ", self.device)
        if self.device:
            self.coords = self.coords.to(self.device)
            self.data = self.data.to(self.device)

    # Pytorch methods
    def __len__(self) -> int:
        """Returns length of dataset. Required by pytorch"""
        return self.number_of_samples

    def __getitem__(self, idx: int) -> int:
        """Returns coordinate and value. First axis of coordinate should be time."""
        return self.coords[idx], self.data[idx]

    # get methods
    def get_coords(self):
        """Retrieve all the coordinate features"""
        return self.coords

    def get_data(self):
        """Retrieve all the data features"""
        return self.data

    # Logical methods
    def preprocess(
        self,
        X: torch.tensor,
        y: torch.tensor,
        random_state: int = 42,
        noise_level: float = 0.0,
        normalize_coords: bool = False,
        normalize_data: bool = False,
    ):
        """Add noise to the data and normalize the features
        Arguments:
            X (torch.tensor) : coordinates of the dataset
            y (torch.tensor) : values of the dataset
            random_state (int) : state for random number geerator
            noise (float) : standard deviations of noise to add
            normalize_coords (bool): apply normalization to the coordinates
            normalize_data (bool): apply normalization to the data
        """

        # add noise
        y_processed = y + self.apply_noise(y, noise_level, random_state)
        # normalize coordinates
        if normalize_coords:
            X_processed = self.apply_normalize(X)
        else:
            X_processed = X
        # normalize data
        if normalize_data:
            y_processed = self.apply_normalize(y_processed)
        else:
            y_processed = y
        return X_processed, y_processed

    @staticmethod
    def apply_noise(y, noise_level, random_state):
        """Adds gaussian white noise of noise_level standard deviation.
        Args:
            y (torch.tensor): the data to which noise should be added
            noise_level (float): add white noise as a function of standard deviation
            random_state (int): the random state used for random number generation
        """
        noise = noise_level * torch.std(y).data
        y_noisy = y + torch.tensor(
            default_rng(random_state).normal(loc=0.0, scale=noise, size=y.shape),
            dtype=torch.float32,
        )  # TO DO: switch to pytorch rng
        return y_noisy

    @staticmethod
    def apply_normalize(X):
        """minmax Normalize the data along the zeroth axis. Per feature
        Args:
            X (torch.tensor): data to be minmax normalized
        Returns:
            (torch.tensor): minmaxed data"""
        X_norm = (X - X.view(-1, X.shape[-1]).min(dim=0).values) / (
            X.view(-1, X.shape[-1]).max(dim=0).values
            - X.view(-1, X.shape[-1]).min(dim=0).values
        ) * 2 - 1
        return X_norm

    @staticmethod
    def apply_shuffle(coords, data):
        """Shuffle the coordinates and data"""
        permutation = np.random.permutation(np.arange(len(data)))
        return coords[permutation], data[permutation]

    @staticmethod
    def _reshape(coords, data):
        """Reshape the coordinates and data to the format [number_of_samples, number_of_features]"""
        coords = coords.reshape([-1, coords.shape[-1]])
        data = data.reshape([-1, data.shape[-1]])
        return coords, data


class Loader:
    def __init__(self, dataset):
        """Loader created to follow the workflow of PyTorch Dataset and Dataloader
        Leaves all data where it currently is."""
        if isinstance(dataset, torch.utils.data.Subset):
            self.device = dataset.dataset.device
        else:
            self.device = dataset.device
        self.dataset = dataset
        self._count = 0
        self._length = 1

    def __getitem__(self, idx):
        if idx < self._length:
            return self.dataset[:]
        else:
            raise StopIteration

    def __len__(self):
        return self._length


def get_train_test_loader(
    dataset, train_test_split=0.8, loader=Loader, loader_kwargs={}
):
    """Take a dataset, shuffle it, split it into a train and test and then
    return two loaders that are compatible with PyTorch.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to use
        train_test_split (float, optional): The fraction of data used for train. Defaults to 0.8.
        loader (torch.utils.data.Dataloader, optional): The type of Dataloader to use. Defaults to GPULoader.
        loader_kwargs (dict, optional): Any kwargs to be passed to the loader]. Defaults to {}.

    Returns:
        Dataloader, Dataloader: The train and test dataloader
    """
    length = dataset.number_of_samples
    indices = np.arange(0, length, dtype=int)
    np.random.shuffle(indices)
    split = int(train_test_split * length)
    train_indices = indices[:split]
    test_indices = indices[split:]
    train_data = torch.utils.data.Subset(dataset, train_indices)
    test_data = torch.utils.data.Subset(dataset, test_indices)
    return loader(train_data, **loader_kwargs), loader(test_data, **loader_kwargs)
