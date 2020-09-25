import torch
'''This module implements convergence criteria'''


class Convergence:
    '''Implements convergence criterium. Convergence is when change in patience
    epochs is smaller than delta.
    '''
    def __init__(self, patience: int = 100, delta: float = 0.05) -> None:
        self.patience = patience
        self.delta = delta
        self.counter: int = 0
        self.start_l1: torch.Tensor = None
        self.converged = False

    def __call__(self, epoch: int, l1_norm: torch.Tensor) -> None:
        """[summary]

        Args:
            epoch (int): [description]
            l1_norm (torch.Tensor): [description]
        """
        if self.start_l1 is None:
            self.start_l1 = l1_norm
        elif torch.abs(self.start_l1 - l1_norm).item() < self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.converged = True
        else:
            self.start_l1 = l1_norm
            self.counter = 0
