import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Tensorboard():
    '''Tensorboard class for logging during deepmod training. '''
    def __init__(self, number_of_terms, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.writer.add_custom_scalars(custom_board(number_of_terms))

    def write(self, iteration, loss, loss_mse, loss_reg, loss_l1,
              constraint_coeff_vectors, unscaled_constraint_coeff_vectors, **kwargs):
        # Costs and coeff vectors
        self.writer.add_scalar('Total loss', loss, iteration)
        for idx in range(len(loss_mse)):
            self.writer.add_scalar(f'MSE {idx}', loss_mse[idx], iteration)
            self.writer.add_scalar(f'Regression {idx}', loss_reg[idx], iteration)
            self.writer.add_scalar(f'L1 {idx}', loss_l1[idx], iteration)
            for element_idx, element in enumerate(torch.unbind(constraint_coeff_vectors[idx])):  # Tensorboard doesnt have vectors, so we unbind and plot them in together in custom board
                self.writer.add_scalar(f'coeff {idx} {element_idx}', element, iteration)
            for element_idx, element in enumerate(torch.unbind(unscaled_constraint_coeff_vectors[idx])):
                self.writer.add_scalar(f'unscaled_coeff {idx} {element_idx}', element, iteration)

        # Writing remaining kwargs
        for key, value in kwargs.items():
            assert len(value.squeeze().shape) <= 1, 'writing matrices is not supported.'
            if len(value.squeeze().shape) == 0:  # if scalar
                self.writer.add_scalar(key, value, iteration)
            else:  # else its a vector and we have to unbind
                for element_idx, element in enumerate(torch.unbind(value)):
                    self.writer.add_scalar(key + f'_{element_idx}', element, iteration)

    def close(self):
        self.writer.close()


def custom_board(number_of_terms):
    '''Custom scalar board for tensorboard.'''
    number_of_eqs = len(number_of_terms)
    # Initial setup, including all the costs and losses
    custom_board = {'Costs': {'MSE': ['Multiline', [f'MSE {idx}' for idx in np.arange(number_of_eqs)]],
                              'Regression': ['Multiline', [f'Regression {idx}' for idx in np.arange(number_of_eqs)]],
                              'L1': ['Multiline', [f'L1 {idx}' for idx in np.arange(number_of_eqs)]]},
                    'Coefficients': {},
                    'Unscaled coefficients': {}}

    # Add plot of normal and scaled coefficients for each equation, containing every component in single plot.
    for idx in np.arange(number_of_eqs):
        custom_board['Coefficients'][f'Vector {idx}'] = ['Multiline', [f'coeff {idx} {element_idx}'
                                                         for element_idx in np.arange(number_of_terms[idx])]]
        custom_board['Unscaled coefficients'][f'Vector {idx}'] = ['Multiline', [f'unscaled_coeff {idx} {element_idx}'
                                                                  for element_idx in np.arange(number_of_terms[idx])]]
    return custom_board
