""" Contains the base class for the Dataset (1 and 2 dimensional) and a function
     that takes a Pytorch tensor and converts it to a numpy array"""

import torch
import numpy as np
from numpy import ndarray
from numpy.random import default_rng
from deepymod.data.samples import Subsampler

from abc import ABC, ABCMeta, abstractmethod


def pytorch_func(function):
    """Decorator to automatically transform arrays to tensors and back

    Args:
        function (Tensor): Pytorch tensor

    Returns:
        (wrapper): function that can evaluate the Pytorch function for extra
        (keyword) arguments, returning (np.array)
    """

    def wrapper(self, *args, **kwargs):
        """Evaluate function, Assign arugments and keyword arguments to a
         Pytorch function and return it as a numpy array.

        Args:
            *args: argument
            **kwargs: keyword arguments
        Return
            (np.array): output of function Tensor converted to numpy array"""
        torch_args = [
            torch.tensor(arg, requires_grad=True, dtype=torch.float64)
            if type(arg) is ndarray
            else arg
            for arg in args
        ]
        torch_kwargs = {
            key: torch.tensor(kwarg, requires_grad=True, dtype=torch.float64)
            if type(kwarg) is ndarray
            else kwarg
            for key, kwarg in kwargs.items()
        }
        result = function(self, *torch_args, **torch_kwargs)
        return result.cpu().detach().numpy()

    return wrapper


class Dataset_old:
    """This class automatically generates all the necessary proporties of a
    predefined data set with a single spatial dimension as input.
    In particular it calculates the solution, the time derivative and the library.
    Note that all the pytorch opperations such as automatic differentiation can be used on the results.
    """

    def __init__(self, solution, **kwargs):
        """Create a dataset and add a solution (data) to it
        Args:
            solution: give the solution, the actual dataset u
            parameters: additional parameters in keyword format
        """
        self.solution = solution  # set solution
        self.parameters = kwargs  # set solution parameters
        self.scaling_factor = None

    @pytorch_func
    def generate_solution(self, x, t):
        """Generates the solution for a set of input coordinates

        Args:
            x (Tensor): Input vector of spatial coordinates
            t (Tensor): Input vector of temporal coordinates

        Returns:
            [Tensor]: Solution evaluated at the input coordinates
        """
        u = self.solution(x, t, **self.parameters)
        return u

    @pytorch_func
    def time_deriv(self, x, t):
        """Generates the time derivative for a set of input coordinates

        Args:
            x (Tensor): Input vector of spatial coordinates
            t (Tensor): Input vector of temporal coordinates

        Returns:
            [Tensor]: Temporal derivate evaluated at the input coordinates
        """
        u = self.solution(x, t, **self.parameters)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u))[0]
        return u_t

    @pytorch_func
    def library(self, x, t, poly_order=2, deriv_order=2):
        """Returns library with given derivative and polynomial order

        Args:
            x (Tensor): Input vector of spatial coordinates
            t (Tensor): Input vector of temporal coordinates
            poly_order (int, optional): Max. polynomial order of the library, defaults to 2.
            deriv_order (int, optional): Max. derivative orderof the library , defaults to 2.

        Returns:
            [Tensor]: Library
        """
        assert (x.shape[1] == 1) & (
            t.shape[1] == 1
        ), "x and t should have shape (n_samples x 1)"

        u = self.solution(x, t, **self.parameters)

        # Polynomial part
        poly_library = torch.ones_like(u)
        for order in torch.arange(1, poly_order + 1):
            poly_library = torch.cat(
                (poly_library, poly_library[:, order - 1 : order] * u), dim=1
            )

        # derivative part
        if deriv_order == 0:
            deriv_library = torch.ones_like(u)
        else:
            du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
            deriv_library = torch.cat((torch.ones_like(u), du), dim=1)
            if deriv_order > 1:
                for order in torch.arange(1, deriv_order):
                    du = torch.autograd.grad(
                        deriv_library[:, order : order + 1],
                        x,
                        grad_outputs=torch.ones_like(u),
                        create_graph=True,
                    )[0]
                    deriv_library = torch.cat((deriv_library, du), dim=1)

        # Making library
        theta = torch.matmul(
            poly_library[:, :, None], deriv_library[:, None, :]
        ).reshape(u.shape[0], -1)
        return theta

    def create_dataset(
        self,
        x,
        t,
        n_samples,
        noise,
        random=True,
        normalize=True,
        return_idx=False,
        random_state=42,
    ):
        """Function creates the data set in the precise format used by DeepMoD

        Args:
            x (Tensor): Input vector of spatial coordinates
            t (Tensor): Input vector of temporal coordinates
            n_samples (int): Number of samples, set n_samples=0 for all.
            noise (float): Noise level in percentage of std.
            random (bool, optional): When true, data set is randomised. Defaults to True.
            normalize (bool, optional): When true, data set is normalized. Defaults to True.
            return_idx (bool, optional): When true, the id of the data, before randomizing is returned. Defaults to False.
            random_state (int, optional): Seed of the randomisation. Defaults to 42.

        Returns:
            [type]: Tensor containing the input and output and optionally the randomisation.
        """
        assert (x.shape[1] == 1) & (
            t.shape[1] == 1
        ), "x and t should have shape (n_samples x 1)"
        u = self.generate_solution(x, t)

        X = np.concatenate([t, x], axis=1)
        if random_state is None:
            y = u + noise * np.std(u, axis=0) * np.random.normal(size=u.shape)
        else:
            y = u + noise * np.std(u, axis=0) * np.random.RandomState(
                seed=random_state
            ).normal(size=u.shape)

        # creating random idx for samples
        N = y.shape[0] if n_samples == 0 else n_samples

        if random is True:
            if random_state is None:
                rand_idx = np.random.permutation(y.shape[0])[:N]
            else:
                rand_idx = np.random.RandomState(seed=random_state).permutation(
                    y.shape[0]
                )[:N]
        else:
            rand_idx = np.arange(y.shape[0])[:N]

        # Normalizing
        if normalize:
            if self.scaling_factor is None:
                self.scaling_factor = (
                    -(np.max(X, axis=0) + np.min(X, axis=0)) / 2,
                    (np.max(X, axis=0) - np.min(X, axis=0)) / 2,
                )  # only calculate the first time
            X = (X + self.scaling_factor[0]) / self.scaling_factor[1]

        # Building dataset
        X_train = torch.tensor(X[rand_idx, :], dtype=torch.float32)
        y_train = torch.tensor(y[rand_idx, :], dtype=torch.float32)

        if return_idx is False:
            return X_train, y_train
        else:
            return X_train, y_train, rand_idx


class Dataset_2D:
    """This class automatically generates all the necessary proporties of a predifined data set with two spatial dimension as input.
    In particular it calculates the solution, the time derivative and the library. Note that all the pytorch opperations such as automatic differentiation can be used on the results.

    """

    def __init__(self, solution, **kwargs):
        """Create a 2D dataset and add a solution (data) to it
                Args:
        solution: give the solution, the actual dataset u
                    parameters: additional parameters in keyword format
        """
        self.solution = solution  # set solution
        self.parameters = kwargs  # set solution parameters

    @pytorch_func
    def generate_solution(self, x, t):
        """Generates the solution for a set of input coordinates

        Args:
            x (Tensor): Input vector of spatial coordinates
            t (Tensor): Input vector of temporal coordinates

        Returns:
            [Tensor]: Solution evaluated at the input coordinates
        """
        u = self.solution(x, t, **self.parameters)
        return u

    @pytorch_func
    def time_deriv(self, x, t):
        """Generates the time derivative for a set of input coordinates

        Args:
            x (Tensor): Input vector of spatial coordinates
            t (Tensor): Input vector of temporal coordinates

        Returns:
            [Tensor]: Temporal derivate evaluated at the input coordinates
        """
        u = self.solution(x, t, **self.parameters)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u))[0]
        return u_t

    @pytorch_func
    def library(self, x, t, poly_order=0):
        """Returns library with and polynomial order and fixed derivative order (1, u_x, u_y, u_xx, u_yy, u_xy)

        Args:
            x (Tensor): Input vector of spatial coordinates
            t (Tensor): Input vector of temporal coordinates
            poly_order (int, optional): Max. polynomial order of the library, defaults to 0.

        Returns:
            [Tensor]: Library
        """
        assert (x.shape[1] == 2) & (
            t.shape[1] == 1
        ), "x and t should have shape (n_samples x 1)"

        u = self.solution(x, t, **self.parameters)

        # Polynomial part
        poly_library = torch.ones_like(u)
        for order in torch.arange(1, poly_order + 1):
            poly_library = torch.cat(
                (poly_library, poly_library[:, order - 1 : order] * u), dim=1
            )

        # derivative part
        if deriv_order == 0:
            deriv_library = torch.ones_like(u)
        else:
            du = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u), create_graph=True
            )[0]

            u_x = du[:, 0:1]
            u_y = du[:, 1:2]
            du2 = torch.autograd.grad(
                u_x, x, grad_outputs=torch.ones_like(u), create_graph=True
            )[0]
            u_xx = du2[:, 0:1]
            u_xy = du2[:, 1:2]
            du2y = torch.autograd.grad(
                u_y, x, grad_outputs=torch.ones_like(u), create_graph=True
            )[0]
            u_yy = du2y[:, 1:2]
            deriv_library = torch.cat(
                (torch.ones_like(u_x), u_x, u_y, u_xx, u_yy, u_xy), dim=1
            )

        # Making library
        theta = torch.matmul(
            poly_library[:, :, None], deriv_library[:, None, :]
        ).reshape(u.shape[0], -1)
        return theta

    def create_dataset(
        self, x, t, n_samples, noise, random=True, return_idx=False, random_state=42
    ):
        """Function creates the data set in the precise format used by DeepMoD

        Args:
            x (Tensor): Input vector of spatial coordinates
            t (Tensor): Input vector of temporal coordinates
            n_samples (int): Number of samples, set n_samples=0 for all.
            noise (float): Noise level in percentage of std.
            random (bool, optional): When true, data set is randomised. Defaults to True.
            normalize (bool, optional): When true, data set is normalized. Defaults to True.
            return_idx (bool, optional): When true, the id of the data, before randomizing is returned. Defaults to False.
            random_state (int, optional): Seed of the randomisation. Defaults to 42.

        Returns:
            [type]: Tensor containing the input and output and optionally the randomisation.
        """
        assert (x.shape[1] == 2) & (
            t.shape[1] == 1
        ), "x and t should have shape (n_samples x 1)"
        u = self.generate_solution(x, t)

        X = np.concatenate([t, x], axis=1)
        y = u + noise * np.std(u, axis=0) * np.random.normal(size=u.shape)

        # creating random idx for samples
        N = y.shape[0] if n_samples == 0 else n_samples

        if random is True:
            rand_idx = np.random.RandomState(seed=random_state).permutation(y.shape[0])[
                :N
            ]  # so we can get similar splits for different noise levels
        else:
            rand_idx = np.arange(y.shape[0])[:N]

        # Building dataset
        X_train = torch.tensor(X[rand_idx, :], requires_grad=True, dtype=torch.float32)
        y_train = torch.tensor(y[rand_idx, :], requires_grad=True, dtype=torch.float32)

        if return_idx is False:
            return X_train, y_train
        else:
            return X_train, y_train, rand_idx


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        load_function,
        subsampler: Subsampler = None,
        subsampler_kwargs: dict = {},
        load_kwargs: dict = {},
        preprocess_kwargs: dict = {"normalize_coords": False, "normalize_data": False},
        device: str = None,
    ):
        """A dataset class that loads the data, preprocesses it and lastly applies subsampling to it
        Args:
            load_function (function): Must return some input/coordinate of shape [N,...] and some output/data of shape [N, ...] as torch tensors
            subsampler (Subsampler): Function that applies some kind of subsampling to it
            load_kwargs (dict): optional arguments for the load method
            preprocess_kwargs (dict): optional arguments for the preprocess method
            subsample_kwargs (dict): optional arguments for the subsample method
            device (string): which device to send the data to
        Returns:
            (torch.utils.data.Dataset)"""
        self.load = load_function
        self.subsampler = subsampler
        self.load_kwargs = load_kwargs
        self.preprocess_kwargs = preprocess_kwargs
        self.subsampler_kwargs = subsampler_kwargs  # so total number of samples is size(self.t_domain) * n_samples_per_frame
        self.device = device
        self.coords = None
        self.data = None
        self.shuffle = True
        self.coords, self.data = self.load(**self.load_kwargs)
        self.number_of_samples = self.data.size(-1)
        self.coords, self.data = self.preprocess(
            self.coords, self.data, **self.preprocess_kwargs
        )
        if self.subsampler:
            self.coords, self.data = self.subsampler.sample(
                self.coords, self.data, **self.subsampler_kwargs
            )
        if self.shuffle:
            self.coords, self.data = self.apply_shuffle(self.coords, self.data)
        self.number_of_samples = self.data.shape[0]

        print("Dataset is using device: ", self.device)
        if self.device:
            self.coords = self.coords.to(self.device)
            self.data = self.data.to(self.device)

    # Pytorch methods
    def __len__(self) -> int:
        """ Returns length of dataset. Required by pytorch"""
        return self.number_of_samples

    def __getitem__(self, idx: int) -> int:
        """ Returns coordinate and value. First axis of coordinate should be time."""
        return self.coords[idx], self.data[idx]

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
        y_processed = y + self.add_noise(y, noise_level, random_state)
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
    def add_noise(y, noise_level, random_state):
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
        """minmax Normalize the data along the zeroth axis.
        Args:
            X (torch.tensor): data to be minmax normalized
        Returns:
            (torch.tensor): minmaxed data"""
        X_norm = (X - X.min(dim=0).values) / (
            X.max(dim=0).values - X.min(dim=0).values
        ) * 2 - 1
        return X_norm

    @staticmethod
    def apply_shuffle(coords, data):
        """ Shuffle the coordinates and data """
        permutation = np.random.permutation(np.arange(len(data)))
        return coords[permutation], data[permutation]


class GPULoader:
    def __init__(self, dataset):
        """Loader created to follow the workflow of PyTorch Dataset and Dataloader"""
        self.device = dataset.dataset.device
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
    dataset, train_test_split=0.8, loader=GPULoader, loader_kwargs={}
):
    length = dataset.number_of_samples
    indices = np.arange(0, length, dtype=int)
    np.random.shuffle(indices)
    split = int(train_test_split * length)
    train_indices = indices[:split]
    test_indices = indices[split:]
    train_data = torch.utils.data.Subset(dataset, train_indices)
    test_data = torch.utils.data.Subset(dataset, test_indices)
    return loader(train_data, **loader_kwargs), loader(test_data, **loader_kwargs)
