from torch.utils.tensorboard import SummaryWriter

class Tensorboard():
    '''Tensorboard class for logging during deepmod training. '''
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir, max_queue=5, flush_secs=10)
        #self.writer.add_custom_scalars(custom_board(number_of_terms))

    def write(self, iteration, loss, loss_mse, loss_reg, loss_l1,
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

    def close(self):
        self.writer.flush()  # flush remaining stuff to disk
        self.writer.close()  # close writer