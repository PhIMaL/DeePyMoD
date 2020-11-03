""" Contains classes that schedule when the sparsity mask should be applied """
import torch
import numpy as np


class Periodic:
    """Periodically applies sparsity every periodicity iterations
    after initial_epoch.
    """
    def __init__(self, periodicity=50, initial_iteration=1000): 
        """Periodically applies sparsity every periodicity iterations
        after initial_epoch.
        Args:
            periodicity (int): after initial_iterations, apply sparsity mask per periodicity epochs
            initial_iteration (int): wait initial_iterations before applying sparsity
        """
        self.periodicity = periodicity
        self.initial_iteration = initial_iteration       

    def __call__(self, iteration, loss, model, optimizer):
        # Update periodically 
        apply_sparsity = False # we overwrite it if we need to update

        if (iteration - self.initial_iteration) % self.periodicity == 0:
            apply_sparsity = True

        return apply_sparsity

class TrainTest:
    """Early stops the training if validation loss doesn't improve after a given patience. 
       Note that periodicity should be multitude of write_iterations."""
    def __init__(self, patience=200, delta=1e-5, path='checkpoint.pt'):
        """Early stops the training if validation loss doesn't improve after a given patience. 
        Note that periodicity should be multitude of write_iterations.
        Args:
            patience (int): wait patience epochs before checking TrainTest
            delta (float): desired accuracy
            path (str): pathname where to store the savepoints, must have ".pt" extension
            """
        self.path = path
        self.patience = patience
        self.delta = delta
       
        self.best_iteration = None
        self.best_loss = None

    def __call__(self, iteration, loss, model, optimizer):
        apply_sparsity = False # we overwrite it if we need to update

        # Initialize if doesnt exist yet
        if self.best_loss is None:
            self.best_loss = loss
            self.best_iteration = iteration
            self.save_checkpoint(model, optimizer)

        # If it didnt improve, check if we're past patience
        elif (self.best_loss - loss) < self.delta:
            if (iteration - self.best_iteration) >= self.patience:
                # We reload the model to the best point and reset the scheduler
                self.load_checkpoint(model, optimizer)  # reload model to best point
                self.best_loss = None
                self.best_iteration = None
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
        '''Loads model from disk'''
        checkpoint_path = self.path + 'checkpoint.pt'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class TrainTestPeriodic:
    """Early stops the training if validation loss doesn't improve after a given patience. 
       Note that periodicity should be multitude of write_iterations."""
    def __init__(self, periodicity=50, patience=200, delta=1e-5, path='checkpoint.pt'):
        """Early stops the training if validation loss doesn't improve after a given patience. 
        Note that periodicity should be multitude of write_iterations.
        Args:
            periodicity (int): apply sparsity mask per periodicity epochs
            patience (int): wait patience epochs before checking TrainTest
            delta (float): desired accuracy
            path (str): pathname where to store the savepoints, must have ".pt" extension"""
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
        '''Loads model from disk'''
        checkpoint_path = self.path + 'checkpoint.pt'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])