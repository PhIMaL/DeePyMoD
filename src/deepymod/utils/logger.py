""" Module to log performance metrics whilst training Deepmyod """
import numpy as np
import torch
import sys, time
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, comment=None, log_dir=None, max_queue=5, flush_secs=10):
            base_dir = Path(log_dir or "runs")
            base_dir.mkdir(parents=True, exist_ok=True)
            self._log_dir = str(base_dir.resolve()) + "/"

        def get_logdir(self):
            return self._log_dir

        def add_scalar(self, *args, **kwargs):
            return None

        def add_scalars(self, *args, **kwargs):
            return None

        def flush(self):
            return None

        def close(self):
            return None


class Logger:
    def __init__(self, exp_ID, log_dir):
        """Log the training process of Deepymod.
        Args:
            exp_ID (str): name or ID of the this experiment
            log_dir (str): directory to save the log files to disk.

        """
        self.writer = SummaryWriter(
            comment=exp_ID, log_dir=log_dir, max_queue=5, flush_secs=10
        )
        self.log_dir = self.writer.get_logdir()

    @staticmethod
    def coefficient_items(values):
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        return np.atleast_1d(np.squeeze(values))

    def __call__(
        self,
        iteration,
        loss,
        MSE,
        Reg,
        constraint_coeffs,
        unscaled_constraint_coeffs,
        estimator_coeffs,
        **kwargs,
    ):
        l1_norm = torch.stack([
            torch.sum(torch.abs(coeffs))
            for coeffs in constraint_coeffs
        ])

        self.update_tensorboard(
            iteration,
            loss,
            MSE,
            Reg,
            l1_norm,
            constraint_coeffs,
            unscaled_constraint_coeffs,
            estimator_coeffs,
            **kwargs,
        )
        self.update_terminal(iteration, MSE, Reg, l1_norm)

    def update_tensorboard(
        self,
        iteration,
        loss,
        loss_mse,
        loss_reg,
        loss_l1,
        constraint_coeff_vectors,
        unscaled_constraint_coeff_vectors,
        estimator_coeff_vectors,
        **kwargs,
    ):
        """Write the current state of training to Tensorboard
        Args:
            iteration (int): iteration number
            loss (float): loss value
            loss_mse (float): loss of the Mean Squared Error term
            loss_reg (float): loss of the regularization term
            loss_l1 (float): loss of the L1 penalty term
            constraint_coeff_vectors (np.array): vector with constraint coefficients
            unscaled_constraint_coeff_vectors (np.array): unscaled vector with constraint coefficients
            estimator_coeff_vectors (np.array): coefficients as computed by the estimator.
        """
        # Costs and coeff vectors
        self.writer.add_scalar("loss/loss", loss, iteration)
        self.writer.add_scalars(
            "loss/mse",
            {f"output_{idx}": val for idx, val in enumerate(loss_mse)},
            iteration,
        )
        self.writer.add_scalars(
            "loss/reg",
            {f"output_{idx}": val for idx, val in enumerate(loss_reg)},
            iteration,
        )
        self.writer.add_scalars(
            "loss/l1",
            {f"output_{idx}": val for idx, val in enumerate(loss_l1)},
            iteration,
        )

        for output_idx, (coeffs, unscaled_coeffs, estimator_coeffs) in enumerate(
            zip(
                constraint_coeff_vectors,
                unscaled_constraint_coeff_vectors,
                estimator_coeff_vectors,
            )
        ):
            self.writer.add_scalars(
                f"coeffs/output_{output_idx}",
                {
                    f"coeff_{idx}": val
                    for idx, val in enumerate(self.coefficient_items(coeffs))
                },
                iteration,
            )
            self.writer.add_scalars(
                f"unscaled_coeffs/output_{output_idx}",
                {
                    f"coeff_{idx}": val
                    for idx, val in enumerate(self.coefficient_items(unscaled_coeffs))
                },
                iteration,
            )
            self.writer.add_scalars(
                f"estimator_coeffs/output_{output_idx}",
                {
                    f"coeff_{idx}": val
                    for idx, val in enumerate(self.coefficient_items(estimator_coeffs))
                },
                iteration,
            )

        # Writing remaining kwargs
        for key, value in kwargs.items():
            if value.numel() == 1:
                self.writer.add_scalar(f"remaining/{key}", value, iteration)
            else:
                self.writer.add_scalars(
                    f"remaining/{key}",
                    {
                        f"val_{idx}": val.squeeze()
                        for idx, val in enumerate(value.squeeze())
                    },
                    iteration,
                )

    def update_terminal(self, iteration, MSE, Reg, L1):
        """Prints and updates progress of training cycle in command line."""
        sys.stdout.write(
            f"\r{iteration:>6}  MSE: {torch.sum(MSE).item():>8.2e}  Reg: {torch.sum(Reg).item():>8.2e}  L1: {torch.sum(L1).item():>8.2e} "
        )
        sys.stdout.flush()

    def close(self, model):
        """Close the Tensorboard writer"""
        print("Algorithm converged. Writing model to disk.")
        self.writer.flush()  # flush remaining stuff to disk
        self.writer.close()  # close writer

        # Save model
        model_path = self.log_dir + "model.pt"
        torch.save(model.state_dict(), model_path)
