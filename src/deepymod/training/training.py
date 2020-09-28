import torch
import time
from math import pi
import numpy as np

from ..utils.tensorboard import Tensorboard
from ..utils.output import progress
from .convergence import Convergence
from ..model.deepmod import DeepMoD
from typing import Optional

def train(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          sparsity_scheduler,
          test = 'mse',
          split: float = 0.8,
          log_dir: Optional[str] = None,
          max_iterations: int = 10000,
          write_iterations: int = 25,
          **convergence_kwargs) -> None:
    """Trains the DeepMoD model. This function automatically splits the data set in a train and test set. 

    Args:
        model (DeepMoD):  A DeepMoD object.
        data (torch.Tensor):  Tensor of shape (n_samples x (n_spatial + 1)) containing the coordinates, first column should be the time coordinate.
        target (torch.Tensor): Tensor of shape (n_samples x n_features) containing the target data.
        optimizer ([type]):  Pytorch optimizer.
        sparsity_scheduler ([type]):  Decides when to update the sparsity mask.
        test (str, optional): Sets what to use for the test loss, by default 'mse'
        split (float, optional):  Fraction of the train set, by default 0.8.
        log_dir (Optional[str], optional): Directory where tensorboard file is written, by default None.
        max_iterations (int, optional): [description]. Max number of epochs , by default 10000.
        write_iterations (int, optional): [description]. Sets how often data is written to tensorboard and checks train loss , by default 25.
    """
    start_time = time.time()
    board = Tensorboard(log_dir)  # initializing tb board

    # Splitting data, assumes data is already randomized
    n_train = int(split * data.shape[0])
    n_test = data.shape[0] - n_train
    data_train, data_test = torch.split(data, [n_train, n_test], dim=0)
    target_train, target_test = torch.split(target, [n_train, n_test], dim=0)
    
    # Training
    convergence = Convergence(**convergence_kwargs)
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in np.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data_train)

        MSE = torch.mean((prediction - target_train)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])
        loss = torch.sum(MSE + Reg) 

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            prediction_test, coordinates = model.func_approx(data_test)
            time_derivs_test, thetas_test = model.library((prediction_test, coordinates))
            with torch.no_grad():
                MSE_test = torch.mean((prediction_test - target_test)**2, dim=0)  # loss per output
                Reg_test = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs_test, thetas_test, model.constraint_coeffs(scaled=False, sparse=True))])
                loss_test = torch.sum(MSE_test + Reg_test) 
            
            # ====================== Logging =======================
            _ = model.sparse_estimator(thetas, time_derivs) # calculating l1 adjusted coeffs but not setting mask
            estimator_coeff_vectors = model.estimator_coeffs()
            l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)), dim=0)
            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())
            board.write(iteration, loss, MSE, Reg, l1_norm, model.constraint_coeffs(sparse=True, scaled=True), model.constraint_coeffs(sparse=True, scaled=False), estimator_coeff_vectors, MSE_test=MSE_test, Reg_test=Reg_test, loss_test=loss_test)
            
            # ================== Sparsity update =============
            # Updating sparsity and or convergence
            #sparsity_scheduler(iteration, l1_norm)
            if iteration % write_iterations == 0:
                if test == 'mse':
                    sparsity_scheduler(iteration, torch.sum(MSE_test), model, optimizer)
                else:
                    sparsity_scheduler(iteration, loss_test, model, optimizer) 
                    
                if sparsity_scheduler.apply_sparsity is True:
                    with torch.no_grad():
                        model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                        sparsity_scheduler.reset()

            # ================= Checking convergence
            convergence(iteration, torch.sum(l1_norm))
            if convergence.converged is True:
                print('Algorithm converged. Stopping training.')
                break
    board.close()
