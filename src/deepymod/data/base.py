""" Contains the base class for the Dataset (1 and 2 dimensional) and a function
     that takes a Pytorch tensor and converts it to a numpy array"""

import torch
import numpy as np
from numpy import ndarray


def pytorch_func(function):
    """Decorator to automatically transform arrays to tensors and back

    Args:
        function (Tensor): Pytorch tensor 

    Returns:
        (wrapper): function that can evaluate the Pytorch function for extra 
        (keyword) arguments, returning (np.array)
    """
    def wrapper(self, *args, **kwargs):
        """ Evaluate function, Assign arugments and keyword arguments to a
         Pytorch function and return it as a numpy array.

        Args: 
            *args: argument
            **kwargs: keyword arguments
        Return
            (np.array): output of function Tensor converted to numpy array"""
        torch_args = [torch.tensor(arg, requires_grad=True, dtype=torch.float64) if type(arg) is ndarray else arg for arg in args]
        torch_kwargs = {key: torch.tensor(kwarg, requires_grad=True, dtype=torch.float64) if type(kwarg) is ndarray else kwarg for key, kwarg in kwargs.items()}
        result = function(self, *torch_args, **torch_kwargs)
        return result.cpu().detach().numpy()
    return wrapper


class Dataset:
    """ This class automatically generates all the necessary proporties of a
    predefined data set with a single spatial dimension as input.
    In particular it calculates the solution, the time derivative and the library.
    Note that all the pytorch opperations such as automatic differentiation can be used on the results. 
 
    """
    def __init__(self, solution, **kwargs):
        """ Create a dataset and add a solution (data) to it
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
        """ Create a 2D dataset and add a solution (data) to it
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

