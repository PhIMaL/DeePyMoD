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
        """TO DO: LOAD MODEL AND OPTIMIZER LIKE TRAINTESTPERIODIC
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
        self.best_iteration = 0
        self.best_score = None
        self.apply_sparsity = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, iteration, val_loss, model, optimizer):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, optimizer)
        elif score < self.best_score + self.delta:
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if (iteration - self.best_iteration) >= self.patience:
                self.apply_sparsity = True
        else:
            self.best_score = score
            self.save_checkpoint(model, optimizer)
            self.best_iteration = iteration

    def save_checkpoint(self, model, optimizer):
        '''Saves model when validation loss decrease.'''
        checkpoint_path = self.path + 'checkpoint.pt'
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

    def reset(self) -> None:
        """[summary]
        """
      
        self.best_iteration = 0
        self.best_score = None
        self.apply_sparsity = False
        self.val_loss_min = np.Inf


class TrainTestPeriodic:
    """Early stops the training if validation loss doesn't improve after a given patience. 
       Note that periodicity should be multitude of write_iterations."""
    def __init__(self, periodicity=50, patience=200, delta=1e-5, path='checkpoint.pt'):
        self.path = path
        self.patience = patience
        self.delta = delta
        self.periodicity = periodicity
       
        self.best_iteration = None
        self.best_loss = None
        self.periodic = False

    def __call__(self, iteration, loss, model, optimizer):
        # Update periodically if we have updated once
        apply_sparsity = False # we overwrite it if we need to update

        if self.periodic is True:
            if (iteration - self.best_iteration) % self.periodicity == 0:
                apply_sparsity = True

        # Check for improvements if we havent updated yet.
        # Initialize if doesnt exist yet
        elif self.best_loss is None:
            self.best_loss = loss
            self.best_iteration = iteration
            self.save_checkpoint(model, optimizer)

        # If it didnt improve, check if we're past patience
        elif (self.best_loss - loss) < self.delta:
            if (iteration - self.best_iteration) >= self.patience:
                self.load_checkpoint(model, optimizer)  # reload model to best point
                self.periodic = True  # switch to periodic regime
                self.best_iteration = iteration  # because the iterator doesnt reset
                apply_sparsity = True
    
        # If not, keep going
        else:
            self.best_loss = loss
            self.best_iteration = iteration
            self.save_checkpoint(model, optimizer)

        return apply_sparsity

    def save_checkpoint(self, model, optimizer):
        '''Saves model when validation loss decrease.'''
        checkpoint_path = self.path + 'checkpoint.pt'
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}, checkpoint_path)

    def load_checkpoint(self, model, optimizer):
        checkpoint_path = self.path + 'checkpoint.pt'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])