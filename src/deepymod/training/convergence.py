import torch
'''This module implements convergence criteria'''


class Convergence:
    '''Implements convergence criterium. Convergence is when change in patience
    epochs is smaller than delta.
    '''
    def __init__(self, patience: int = 200, delta: float = 1e-3) -> None:
        self.patience = patience
        self.delta = delta
        self.best_iteration = 0
        self.start_l1: torch.Tensor = None
        self.converged = False

    def __call__(self, epoch: int, l1_norm: torch.Tensor) -> None:
        """

        Args:
            epoch (int): Current epoch of the optimization
            l1_norm (torch.Tensor): Value of the L1 norm
        """
        if self.start_l1 is None:
            self.start_l1 = l1_norm
        elif torch.abs(self.start_l1 - l1_norm).item() < self.delta:
            if (epoch - self.best_iteration) >= self.patience:
                self.converged = True
        else:
            self.start_l1 = l1_norm
            self.best_iteration = epoch
