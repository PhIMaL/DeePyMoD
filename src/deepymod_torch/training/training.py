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
          log_dir: Optional[str] = None,
          max_iterations: int = 10000,
          **convergence_kwargs) -> None:
    """[summary]

    Args:
        model (DeepMoD): [description]
        data (torch.Tensor): [description]
        target (torch.Tensor): [description]
        optimizer ([type]): [description]
        sparsity_scheduler ([type]): [description]
        log_dir (Optional[str], optional): [description]. Defaults to None.
        max_iterations (int, optional): [description]. Defaults to 10000.
    """
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_terms, log_dir)  # initializing custom tb board

    # Training
    convergence = Convergence(**convergence_kwargs)
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in np.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, sparse_thetas, thetas, constraint_coeffs = model(data)

        MSE = torch.mean((prediction - target)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs, sparse_thetas, constraint_coeffs)])
        loss = torch.sum(2 * torch.log(2 * pi * MSE) + Reg / (MSE + 1e-6))  # 1e-5 for numerical stability

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ================== Validation and sparsity =============
        # We calculate the normalization factor and the l1_norm
        coeff_norms = [(torch.norm(time_deriv) / torch.norm(theta, dim=0, keepdim=True)).detach().squeeze() for time_deriv, theta in zip(time_derivs, sparse_thetas)]
        l1_norm = torch.stack([torch.sum(torch.abs(coeff_vector.squeeze() / norm)) for coeff_vector, norm in zip(constraint_coeffs, coeff_norms)])

        # Updating sparsity and or convergence
        sparsity_scheduler(iteration, torch.sum(l1_norm))
        if sparsity_scheduler.apply_sparsity is True:
            with torch.no_grad():
                model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                sparsity_scheduler.reset()
                print(model.constraint.sparsity_masks)

        # Checking convergence
        convergence(iteration, torch.sum(l1_norm))
        if convergence.converged is True:
            print('Algorithm converged. Stopping training.')
            break

    
        # ====================== Logging =======================
        # Write progress to command line and tensorboard
        if iteration % 25 == 0:
            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())

            # We pad the sparse vectors with zeros so they get written correctly)
            unscaled_constraint_coeff_vectors = [torch.zeros(mask.size()).masked_scatter_(mask, coeff_vector.detach().squeeze())
                                                 for mask, coeff_vector
                                                 in zip(model.constraint.sparsity_masks, constraint_coeffs)]
            constraint_coeff_vectors = [torch.zeros(mask.size()).masked_scatter_(mask, coeff_vector.detach().squeeze() / norm)
                                        for mask, coeff_vector, norm
                                        in zip(model.constraint.sparsity_masks, constraint_coeffs, coeff_norms)]

            board.write(iteration, loss, MSE, Reg, l1_norm, constraint_coeff_vectors, unscaled_constraint_coeff_vectors)

    board.close()
