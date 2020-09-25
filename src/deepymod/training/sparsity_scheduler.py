import torch
import numpy as np

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



class TrainTest:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.apply_sparsity = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.apply_sparsity = True
        else:
            self.best_score = score
            self.save_checkpoint(model, optimizer)
            self.counter = 0

    def save_checkpoint(self, model, optimizer):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}, self.path)

    def reset(self) -> None:
        """[summary]
        """
      
        self.counter = 0
        self.best_score = None
        self.apply_sparsity = False
        self.val_loss_min = np.Inf

class TrainTestPeriodic:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, periodicity=50, patience=7, delta=0.00, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.apply_sparsity = False
        self.val_loss_min = np.Inf
        self.path = path
        self.initial_epoch = None
        self.periodicity = periodicity
        self.delta = delta

    def __call__(self, iteration, val_loss, model, optimizer):
        score = -val_loss
        if self.initial_epoch is not None:
            if (iteration - self.initial_epoch) % self.periodicity == 0:
                self.apply_sparsity = True 
    
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.apply_sparsity = True
                self.initial_epoch = iteration
                checkpoint = torch.load(self.path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        else:
            self.best_score = score
            self.save_checkpoint(model, optimizer)
            self.counter = 0

    def save_checkpoint(self, model, optimizer):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}, self.path)

    def reset(self) -> None:
        """[summary]
        """
      
        self.counter = 0
        self.best_score = None
        self.apply_sparsity = False
        self.val_loss_min = np.Inf