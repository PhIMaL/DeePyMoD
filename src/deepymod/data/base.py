import torch
import numpy as np
from numpy import ndarray


def pytorch_func(function):
    """Decorator to automatically transform arrays to tensors and back

    Args:
        function (Tensor): Pytorch tensor 

    Returns:
        Numpy array 
    """
    def wrapper(self, *args, **kwargs):
        torch_args = [torch.tensor(arg, requires_grad=True, dtype=torch.float64) if type(arg) is ndarray else arg for arg in args]
        torch_kwargs = {key: torch.tensor(kwarg, requires_grad=True, dtype=torch.float64) if type(kwarg) is ndarray else kwarg for key, kwarg in kwargs.items()}
        result = function(self, *torch_args, **torch_kwargs)
        return result.cpu().detach().numpy()
    return wrapper


class Dataset_old:
    """ This class automatically generates all the necessary proporties of a predifined data set with a single spatial dimension as input.
    In particular it calculates the solution, the time derivative and the library. Note that all the pytorch opperations such as automatic differentiation can be used on the results. 
 
    """
    def __init__(self, solution, **kwargs):
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
        """ Returns library with given derivative and polynomial order

        Args:
            x (Tensor): Input vector of spatial coordinates
            t (Tensor): Input vector of temporal coordinates
            poly_order (int, optional): Max. polynomial order of the library, defaults to 2.
            deriv_order (int, optional): Max. derivative orderof the library , defaults to 2.

        Returns:
            [Tensor]: Library 
        """
        assert ((x.shape[1] == 1) & (t.shape[1] == 1)), 'x and t should have shape (n_samples x 1)'

        u = self.solution(x, t, **self.parameters)

        # Polynomial part
        poly_library = torch.ones_like(u)
        for order in torch.arange(1, poly_order+1):
            poly_library = torch.cat((poly_library, poly_library[:, order-1:order] * u), dim=1)

        # derivative part
        if deriv_order == 0:
            deriv_library = torch.ones_like(u)
        else:
            du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
            deriv_library = torch.cat((torch.ones_like(u), du), dim=1)
            if deriv_order > 1:
                for order in torch.arange(1, deriv_order):
                    du = torch.autograd.grad(deriv_library[:, order:order+1], x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                    deriv_library = torch.cat((deriv_library, du), dim=1)

        # Making library
        theta = torch.matmul(poly_library[:, :, None], deriv_library[:, None, :]).reshape(u.shape[0], -1)
        return theta

    def create_dataset(self, x, t, n_samples, noise, random=True, normalize=True, return_idx=False, random_state=42):
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
        assert ((x.shape[1] == 1) & (t.shape[1] == 1)), 'x and t should have shape (n_samples x 1)'
        u = self.generate_solution(x, t)

        X = np.concatenate([t, x], axis=1)
        if random_state is None:
            y = u + noise * np.std(u, axis=0) * np.random.normal(size=u.shape)
        else:
            y = u + noise * np.std(u, axis=0) *  np.random.RandomState(seed=random_state).normal(size=u.shape)
           

        # creating random idx for samples
        N = y.shape[0] if n_samples == 0 else n_samples

        if random is True:
            if random_state is None:
                rand_idx = np.random.permutation(y.shape[0])[:N]
            else:
                rand_idx = np.random.RandomState(seed=random_state).permutation(y.shape[0])[:N]
        else:
            rand_idx = np.arange(y.shape[0])[:N]
        
        # Normalizing
        if normalize:
            if (self.scaling_factor is None):
                self.scaling_factor = (-(np.max(X, axis=0) + np.min(X, axis=0))/2, (np.max(X, axis=0) - np.min(X, axis=0))/2) # only calculate the first time
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
        """ Returns library with and polynomial order and fixed derivative order (1, u_x, u_y, u_xx, u_yy, u_xy)

        Args:
            x (Tensor): Input vector of spatial coordinates
            t (Tensor): Input vector of temporal coordinates
            poly_order (int, optional): Max. polynomial order of the library, defaults to 0.

        Returns:
            [Tensor]: Library 
        """
        assert ((x.shape[1] == 2) & (t.shape[1] == 1)), 'x and t should have shape (n_samples x 1)'

        u = self.solution(x, t, **self.parameters)

        # Polynomial part
        poly_library = torch.ones_like(u)
        for order in torch.arange(1, poly_order+1):
            poly_library = torch.cat((poly_library, poly_library[:, order-1:order] * u), dim=1)

        # derivative part
        if deriv_order == 0:
            deriv_library = torch.ones_like(u)
        else:            
            du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

            u_x = du[:, 0:1]
            u_y = du[:, 1:2]
            du2 = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_xx = du2[:, 0:1]
            u_xy = du2[:, 1:2]
            du2y = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_yy = du2y[:, 1:2]
            deriv_library = torch.cat((torch.ones_like(u_x), u_x, u_y , u_xx, u_yy, u_xy), dim=1)


        # Making library
        theta = torch.matmul(poly_library[:, :, None], deriv_library[:, None, :]).reshape(u.shape[0], -1)
        return theta

    def create_dataset(self, x, t, n_samples, noise, random=True, return_idx=False, random_state=42):
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
        assert ((x.shape[1] == 2) & (t.shape[1] == 1)), 'x and t should have shape (n_samples x 1)'
        u = self.generate_solution(x, t)

        X = np.concatenate([t, x], axis=1)
        y = u + noise * np.std(u, axis=0) * np.random.normal(size=u.shape)

        # creating random idx for samples
        N = y.shape[0] if n_samples == 0 else n_samples

        if random is True:
            rand_idx = np.random.RandomState(seed=random_state).permutation(y.shape[0])[:N] # so we can get similar splits for different noise levels
        else:
            rand_idx = np.arange(y.shape[0])[:N]

        # Building dataset
        X_train = torch.tensor(X[rand_idx, :], requires_grad=True, dtype=torch.float32)
        y_train = torch.tensor(y[rand_idx, :], requires_grad=True, dtype=torch.float32)
        
        if return_idx is False:
            return X_train, y_train
        else:
            return X_train, y_train, rand_idx

from numpy.random import default_rng


class Dataset:
    def __init__(self, dataset, noise, n_samples_per_frame, device='cpu', normalize=False, randomize=True, split=0.8, random_state=42):
        self.dataset = dataset # function which returns the data and the grid
        self.noise = noise
        self.n_samples = n_samples_per_frame  # so total number of samples is size(self.t_domain) * n_samples_per_frame
        self.device = device
        self._normalize = normalize
        self._randomize = randomize
        self._split = split
        self.random_state = random_state

    # User facing methods
    def __call__(self):
        # Get dataset
        X, y = self.create_dataset(self.n_samples, self.dataset, self.noise, self._normalize, self._randomize, self.random_state)
        trainset, testset = self.prepare_dataset(X, y, self.device, self._split)
        return trainset, testset

    def ground_truth(self, subsampled=False, normalized=False):
        if subsampled:
            n_samples = self.n_samples
        else:
            n_samples = self.dataset()[0].shape[2]

        X, y = self.create_dataset(n_samples, self.dataset, 0.0, normalized, False, self.random_state)        
        return X.reshape(2, -1, n_samples), y.reshape(1, -1, n_samples)

    def grid(self, subsampled=False, normalized=False):
        grid = self.ground_truth(subsampled=subsampled, normalized=normalized)[0]
        X = grid.reshape(-1, 2).to(self.device)
        return X

    @classmethod
    def create_dataset(cls, n_samples, dataset, noise, normalize, randomize, random_state):
        # Get samples
        data, grid = dataset()
        X, y = cls.sample(data, grid, n_samples)  # subsample data and grid
      
        # add noise
        y += cls.add_noise(y, noise, random_state)

        # normalize
        if normalize:
            X = cls.normalize(X)

        # Randomize data
        if randomize:
            X, y = cls.randomize(X, y, random_state)

        return X, y

    @classmethod
    def prepare_dataset(cls, X, y, device, split):
        # Split into test / train, assumes data is randomized
        X_train, y_train, X_test, y_test = cls.split(X, y, split)

        # Move to device
        X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)  

        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def sample(data, grid, n_samples):
        # getting indices of samples
        x_idx = torch.round(torch.linspace(0, grid.shape[2]-1, n_samples)).type(torch.long) # getting x locations

        # getting sample locations from indices
        X = grid[:, :, x_idx].reshape(-1, 2)
        y = data[:, :, x_idx].reshape(-1, 1)

        return X, y

    @staticmethod
    def add_noise(y, noise_level, random_state):
        ''' Adds gaussian white noise'''
        noise = noise_level * torch.std(y).data
        y_noisy = y + torch.tensor(default_rng(random_state).normal(loc=0.0, scale=noise, size=y.shape), dtype=torch.float32) # TO DO: switch to pytorch rng

        return y_noisy

    @staticmethod
    def normalize(X):
        X_norm = (X - X.min(dim=0).values) / (X.max(dim=0).values - X.min(dim=0).values) * 2 - 1
        return X_norm

    @staticmethod
    def randomize(X, y, random_state):
        rand_idx = default_rng(random_state).permutation(X.shape[0]) # TO DO: switch to pytorch rng
        X_rand, y_rand = X[rand_idx, :], y[rand_idx, :]

        return X_rand, y_rand

    @staticmethod
    def split(X, y, split):
        n_train = int(split * X.shape[0])
        n_test = int(X.shape[0] - n_train)
        X_train, X_test = torch.split(X, [n_train, n_test], dim=0)
        y_train, y_test = torch.split(y, [n_train, n_test], dim=0)

        return X_train, y_train, X_test, y_test

