'''This module implements convergence criteria'''
import torch


class Convergence:
    '''Implements convergence criterium. Convergence is when change in patience
    epochs is smaller than delta.
    '''
    def __init__(self, patience: int = 200, delta: float = 1e-3) -> None:
        '''Implements convergence criterium. Convergence is when change in patience
        epochs is smaller than delta.
        Args: 
            patience (int): how often to check for convergence
            delta (float): desired accuracy
        '''
        self.patience = patience
        self.delta = delta
        self.start_iteration = None
        self.start_l1 = None

    def __call__(self, iteration: int, l1_norm: torch.Tensor) -> bool:
        """

        Args:
            epoch (int): Current epoch of the optimization
            l1_norm (torch.Tensor): Value of the L1 norm
        """
        converged = False # overwrite later

        # Initialize if doesn't exist
        if self.start_l1 is None:
            self.start_l1 = l1_norm
            self.start_iteration = iteration
        
        # Check if change is smaller than delta and if we've exceeded patience
        elif torch.abs(self.start_l1 - l1_norm).item() < self.delta:
            if (iteration - self.start_iteration) >= self.patience:
                converged = True

        # If not, reset and keep going
        else:
            self.start_l1 = l1_norm
            self.start_iteration = iteration

        return converged
