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
        shuffle=True,
        preprocess_functions: dict = {"apply_normalize": None, "add_noise": None},
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
            preprocess_functions (dict, optional): override the default normalization and noise addition. Defaults to {"apply_normalize": None, "add_noise": None}.
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
        if (
            "apply_normalize" in preprocess_functions
            and preprocess_functions["apply_normalize"] != None
        ):
            self.apply_normalize = preprocess_functions["apply_normalize"]
        if (
            "add_noise" in preprocess_functions
            and preprocess_functions["add_noise"] != None
        ):
            self.apply_normalize = preprocess_functions["add_noise"]
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
        """ Returns length of dataset. Required by pytorch"""
        return self.number_of_samples

    def __getitem__(self, idx: int) -> int:
        """ Returns coordinate and value. First axis of coordinate should be time."""
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
        """ Shuffle the coordinates and data """
        permutation = np.random.permutation(np.arange(len(data)))
        return coords[permutation], data[permutation]

    @staticmethod
    def _reshape(coords, data):
        """Reshape the coordinates and data to the format [number_of_samples, number_of_features]"""
        coords = coords.reshape([-1, coords.shape[-1]])
        data = data.reshape([-1, data.shape[-1]])
        return coords, data


class GPULoader:
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
    dataset, train_test_split=0.8, loader=GPULoader, loader_kwargs={}
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
