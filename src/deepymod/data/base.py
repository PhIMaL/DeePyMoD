from operator import sub
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

from numpy.random import default_rng, random


class Dataset:
    def __init__(self, data, t_domain, x_domain, noise, n_samples_per_frame, device='cpu', normalize=False, randomize=True, split=0.8, random_state=42):
        '''
        First dimension of data should be time!
        '''
        self.temporal_domain = torch.tensor(t_domain, dtype=torch.float32)
        self.spatial_domain = torch.tensor(x_domain, dtype=torch.float32)

        if callable(data): # if its a function, we run it to get the datya
            t_grid, x_grid = torch.meshgrid(self.temporal_domain, self.spatial_domain)
            self.data = data(x_grid, t_grid)
            self.analytical_solution = data
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            if isinstance(data, np.ndarray):  # if its an array, we tensorize it
                self.data = torch.tensor(data)
            else:
                self.data = data
            self.analytical_solution = None
        else:
            print('Format not recognized. Please supply a function, array or tensor')

        if data.shape[0] != t_domain.shape[0]:
            print('First dimension of data does not have the same shape as time domain.')
        
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
        X, y = self.create_dataset(self.n_samples, self.data, self.temporal_domain, self.spatial_domain, self.noise, self._normalize, self._randomize, self.random_state)
        trainset, testset = self.prepare_dataset(X, y, self.device, self._split)
        return trainset, testset

    def ground_truth(self, subsampled=False, normalized=False):
        if subsampled:
            n_samples = self.n_samples
        else:
            n_samples = self.spatial_domain.shape[0] 

        X, y = self.create_dataset(n_samples, self.data, self.temporal_domain, self.spatial_domain, 0.0, normalized, False, self.random_state)        
        return X.reshape(-1, n_samples, 2), y.reshape(-1, n_samples)

    def grid(self, subsampled=False, normalized=False):
        grid = self.ground_truth(subsampled=subsampled, normalized=normalized)[0]
        X = grid.reshape(-1, 2).to(self.device)
        return X

    # Logic
    def create_dataset(self, n_samples, data, temporal_domain, spatial_domain, noise, normalize, randomize, random_state):
        # Get samples
        X, idx = self.sample_idx(n_samples, temporal_domain, spatial_domain)  # get sample locations and indices
        y = self.sample(data, idx)  # turns indices into values
      
        # add noise
        y = self.add_noise(y, noise, random_state)

        # normalize
        if normalize:
            X = self.normalize(X)

        # Randomize data
        if randomize:
            X, y = self.randomize(X, y, random_state)

        return X, y

    def prepare_dataset(self, X, y, device, split):
        # Split into test / train, assumes data is randomized
        X_train, y_train, X_test, y_test = self.split(X, y, split)

        # Make into tensor and move to device
        X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)  

        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def sample_idx(n_samples, temporal_domain, spatial_domain):
        # getting indices of samples
        x_idx = torch.round(torch.linspace(0, spatial_domain.shape[0]-1, n_samples)).type(torch.long) # getting x locations
        t_idx = torch.linspace(0, temporal_domain.shape[0]-1, temporal_domain.shape[0]).type(torch.long)
        t_grid, x_grid = torch.meshgrid(t_idx, x_idx)

        # getting sample locations from indices
        X = torch.stack(torch.meshgrid(temporal_domain[t_idx], spatial_domain[x_idx]), dim=-1).reshape(-1, 2)
   
        return X, (t_grid.flatten(), x_grid.flatten())

    @staticmethod
    def sample(data, idx):
        return data[idx[0], idx[1]][:, None]

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


class SamplingDataset(Dataset):
    # Logic
    def sample_locations(self, n_samples, method, sampling_time, **sample_kwargs):
        if method == 'grid':  # regular sampling immediately gives sampling locations
            x_samples, t_samples = self.grid_sample(n_samples, self.sampling_domain, sampling_time)

        else:  # irregular sampling returns a sampling distribution
            if method == 'magnitude':
                y = self.dataset.generate_solution(x_grid, t_grid)
                sample_dist = self.magnitude_sample(y)

            elif method == 'dt':
                dydt = self.dataset.time_deriv(x_grid, t_grid)
                sample_dist = self.dt_sample(dydt) 

            elif method == 'library':
                theta = self.dataset.library(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1), **sample_kwargs)
                weights = np.ones(theta.shape[1])
                sample_dist = self._library_sample(theta.reshape(*x_grid.shape, -1), weights) 

            # Generating samples from sampling distribution
            x_samples = np.stack([np.random.choice(self.sampling_domain, size=n_samples, p=sample_dist[:, frame]) for frame in np.arange(sample_dist.shape[1])], axis=1)
            t_samples = np.ones_like(x_samples) * sampling_time
        
        X = np.concatenate((t_samples, x_samples))
        return X

    def sample(self, sample_locs):
        y = self.data[sample_locs]
        return y

    def grid_sample(self, n_samples, sampling_domain, sampling_time, epsilon=1e-3):
            """Samples n_samples per t step in time on sampling domain on a grid.
            Returns locations where to sample."""

            x = np.linspace(sampling_domain.min(), sampling_domain.max(), n_samples)
            x_samples, t_samples = np.meshgrid(x, sampling_time, indexing='ij')

            return x_samples, t_samples

    def magnitude_sample(self, y, epsilon=1e-3):
        """Samples n_samples per t step in time on sampling domain using magnitude sampling,
        i.e. p(u). Epsilon sets how much to sample empty space. Returns locations where to sample."""

        p = np.abs(y) + epsilon
        p = p / np.sum(p, axis=0)

        return  p

    def dt_sample(self, dydt, epsilon=1e-3):
        """Samples n_samples per t step in time on sampling domain using magnitude sampling,
        i.e. p(u). Epsilon sets how much to sample empty space. Returns locations where to sample."""

        p = np.abs(dydt) + epsilon
        p = p / np.sum(p, axis=0)
        return p
    
    def library_sample(self, theta, weights, epsilon=1e-3):
        """Samples n_samples per t step in time on sampling domain using library sampling,
        i.e. p(u). Epsilon sets how much to sample empty space. Returns locations where to sample."""

        p = np.abs(theta) + epsilon
        p = p / np.sum(p, axis=0)  # normalize each term each time step
        p = np.average(p, weights=weights, axis=2)  # combine all terms

        return p