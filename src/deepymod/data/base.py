import torch
import numpy as np
from numpy import ndarray


def pytorch_func(function):
    '''Decorator to automatically transform arrays to tensors and back'''
    def wrapper(self, *args, **kwargs):
        torch_args = [torch.tensor(arg, requires_grad=True, dtype=torch.float64) if type(arg) is ndarray else arg for arg in args]
        torch_kwargs = {key: torch.tensor(kwarg, requires_grad=True, dtype=torch.float64) if type(kwarg) is ndarray else kwarg for key, kwarg in kwargs.items()}
        result = function(self, *torch_args, **torch_kwargs)
        return result.cpu().detach().numpy()
    return wrapper


class Dataset:
    def __init__(self, solution, **kwargs):
        self.solution = solution  # set solution
        self.parameters = kwargs  # set solution parameters
        self.scaling_factor = None

    @pytorch_func
    def generate_solution(self, x, t):
        '''Generation solution.'''
        u = self.solution(x, t, **self.parameters)
        return u

    @pytorch_func
    def time_deriv(self, x, t):
        u = self.solution(x, t, **self.parameters)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u))[0]
        return u_t

    @pytorch_func
    def library(self, x, t, poly_order=2, deriv_order=2):
        ''' Returns library with 3rd order derivs and 2nd order polynomial'''
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

    def create_dataset(self, x, t, n_samples, noise, random=True, normalize=True, return_idx=False, random_state=None):
        ''' Creates dataset for deepmod. set n_samples=0 for all, noise is percentage of std. '''
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
    def __init__(self, solution, **kwargs):
        self.solution = solution  # set solution
        self.parameters = kwargs  # set solution parameters

    @pytorch_func
    def generate_solution(self, x, t):
        '''Generation solution.'''
        u = self.solution(x, t, **self.parameters)
        return u

    @pytorch_func
    def time_deriv(self, x, t):
        u = self.solution(x, t, **self.parameters)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u))[0]
        return u_t

    @pytorch_func
    def library(self, x, t, poly_order=0, deriv_order=2):
        ''' Returns library with 3rd order derivs and 2nd order polynomial'''
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
        ''' Creates dataset for deepmod. set n_samples=0 for all, noise is percentage of std. Random state can
        be used to generate similar sets for different noise levels'''
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

