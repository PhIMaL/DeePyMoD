import numpy as np
import torch
import sys, time
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir, max_queue=5, flush_secs=10)
        self.log_dir = self.writer.get_logdir()

    def __call__(self, iteration, MSE, Reg, l1_norm, **kwargs):
        #self.update_tensorboard()
        self.update_terminal(iteration, MSE, Reg, l1_norm)

    def update_tensorboard(self, iteration, loss, loss_mse, loss_reg, loss_l1,
              constraint_coeff_vectors, unscaled_constraint_coeff_vectors, estimator_coeff_vectors, **kwargs):
        # Costs and coeff vectors
        self.writer.add_scalar('loss/loss', loss, iteration)
        self.writer.add_scalars('loss/mse', {f'output_{idx}': val for idx, val in enumerate(loss_mse)}, iteration)
        self.writer.add_scalars('loss/reg', {f'output_{idx}': val for idx, val in enumerate(loss_reg)}, iteration)
        self.writer.add_scalars('loss/l1', {f'output_{idx}': val for idx, val in enumerate(loss_l1)}, iteration)

        for output_idx, (coeffs, unscaled_coeffs, estimator_coeffs) in enumerate(zip(constraint_coeff_vectors, unscaled_constraint_coeff_vectors, estimator_coeff_vectors)):
            self.writer.add_scalars(f'coeffs/output_{output_idx}', {f'coeff_{idx}': val for idx, val in enumerate(coeffs.squeeze())}, iteration)
            self.writer.add_scalars(f'unscaled_coeffs/output_{output_idx}', {f'coeff_{idx}': val for idx, val in enumerate(unscaled_coeffs.squeeze())}, iteration)
            self.writer.add_scalars(f'estimator_coeffs/output_{output_idx}', {f'coeff_{idx}': val for idx, val in enumerate(estimator_coeffs.squeeze())}, iteration)

        # Writing remaining kwargs
        for key, value in kwargs.items():
            if value.numel() == 1:
                self.writer.add_scalar(f'remaining/{key}', value, iteration)
            else:
                self.writer.add_scalars(f'remaining/{key}', {f'val_{idx}': val.squeeze() for idx, val in enumerate(value.squeeze())}, iteration)

    def update_terminal(self, iteration, MSE, Reg, L1):
        '''Prints and updates progress of training cycle in command line.'''
        sys.stdout.write(f"\r{iteration:>6}  MSE: {torch.sum(MSE).item():>8.2e}  Reg: {torch.sum(Reg).item():>8.2e}  L1: {L1.item():>8.2e} ")
        sys.stdout.flush()

    def close(self, model, optimizer):
        self.writer.flush()  # flush remaining stuff to disk
        self.writer.close()  # close writer
        model_path = self.log_dir + 'model.pt'
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_path)

