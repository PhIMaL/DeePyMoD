import numpy as np
import torch

from deepymod.data.base import Dataset
from deepymod.model.deepmod import DeepMoD, Estimator
from deepymod.model.sparse_estimators import PDEFIND
from deepymod.utils.logger import Logger


class ScalarEstimator(Estimator):
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.float64(2.0)


class MatrixEstimator(Estimator):
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array([[1.0], [0.0]])


def test_apply_normalize_accepts_non_contiguous_tensors():
    values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).permute(1, 0, 2)

    normalized = Dataset.apply_normalize(values)
    flat = normalized.reshape(-1, normalized.shape[-1])

    assert normalized.shape == values.shape
    assert torch.allclose(flat.min(dim=0).values, -torch.ones(values.shape[-1]))
    assert torch.allclose(flat.max(dim=0).values, torch.ones(values.shape[-1]))


def test_estimator_keeps_single_coefficient_and_mask_one_dimensional():
    theta = torch.ones((5, 1))
    time_deriv = 2.0 * torch.ones((5, 1))

    estimator = ScalarEstimator()
    masks = estimator([theta], [time_deriv])

    assert estimator.coeff_vectors[0].shape == (1, 1)
    assert masks[0].shape == (1,)
    assert masks[0].dtype == torch.bool


def test_estimator_flattens_matrix_coefficients_without_extra_dimension():
    theta = torch.ones((5, 2))
    time_deriv = torch.ones((5, 1))

    estimator = MatrixEstimator()
    masks = estimator([theta], [time_deriv])

    assert estimator.coeff_vectors[0].shape == (2, 1)
    assert masks[0].shape == (2,)


def test_constraint_coeffs_scaled_accepts_single_term_norms():
    model = DeepMoD.__new__(DeepMoD)
    model.constraint = type(
        "ConstraintStub",
        (),
        {
            "coeff_vectors": [torch.tensor([[4.0]])],
            "sparsity_masks": [torch.tensor([True])],
        },
    )()
    model.library = type("LibraryStub", (), {"norms": [torch.tensor(2.0)]})()

    coeffs = model.constraint_coeffs(scaled=True)

    assert coeffs[0].shape == (1, 1)
    assert torch.allclose(coeffs[0], torch.tensor([[2.0]]))


def test_logger_handles_coefficients_with_single_or_heterogeneous_lengths():
    assert Logger.coefficient_items(torch.tensor([[1.0]])).shape == (1,)
    assert Logger.coefficient_items(np.array([[1.0], [2.0]])).shape == (2,)

    logger = Logger.__new__(Logger)
    captured = {}

    def capture_tensorboard(*args, **kwargs):
        captured["l1"] = args[4]

    logger.update_tensorboard = capture_tensorboard
    logger.update_terminal = lambda *args, **kwargs: None

    coeffs = [torch.ones((2, 1)), torch.ones((3, 1))]
    logger(
        iteration=0,
        loss=torch.tensor(0.0),
        MSE=torch.tensor([0.0, 0.0]),
        Reg=torch.tensor([0.0, 0.0]),
        constraint_coeffs=coeffs,
        unscaled_constraint_coeffs=coeffs,
        estimator_coeffs=coeffs,
    )

    assert torch.allclose(captured["l1"], torch.tensor([2.0, 3.0]))


def test_pdefind_stlsq_runs_with_installed_pysindy_signature():
    x = np.linspace(-1.0, 1.0, 30)
    X = np.column_stack([x, np.ones_like(x)])
    y = (2.0 * x)[:, None]

    coeffs = PDEFIND.TrainSTLSQ(X, y, alpha=0.0, delta_threshold=0.01, max_iterations=2)

    assert np.asarray(coeffs).reshape(-1).shape == (2,)
