import torch


class Periodic:
    '''Controls when to apply sparsity. Initial_epoch is first time of appliance,
    then every periodicity epochs.
    '''
    def __init__(self, initial_epoch: int = 1000, periodicity: int = 100) -> None:
        self.initial_epoch = initial_epoch
        self.periodicity = periodicity

        self.apply_sparsity = False

    def __call__(self, iteration: int, l1_norm: torch.Tensor) -> None:
        """[summary]

        Args:
            iteration (int): [description]
            l1_norm (torch.Tensor): [description]
        """
        if iteration >= self.initial_epoch:
            if (iteration - self.initial_epoch) % self.periodicity == 0:
                self.apply_sparsity = True

    def reset(self) -> None:
        """[summary]
        """
        self.apply_sparsity = False
