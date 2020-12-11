# %% Imports
import numpy as np
import torch

from sklearn.linear_model import Ridge as RidgeReference
from deepymod.model.constraint import Ridge

from deepymod.data import Dataset
from deepymod.data.burgers import BurgersDelta

# %% Making dataset
v = 0.1
A = 1.0

x = np.linspace(-3, 4, 100)
t = np.linspace(0.5, 5.0, 50)
x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
dataset = Dataset(BurgersDelta, v=v, A=A)

theta = dataset.library(
    x_grid.reshape(-1, 1), t_grid.reshape(-1, 1), poly_order=2, deriv_order=3
)
dt = dataset.time_deriv(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1))

theta = theta / np.linalg.norm(theta, axis=0, keepdims=True)
dt += 0.1 * np.random.randn(*dt.shape)

# %% Baseline
reg = RidgeReference(alpha=1e-3, fit_intercept=False)
coeff_ref = reg.fit(theta, dt.squeeze()).coef_[:, None]
# %%
constraint = Ridge(l=1e-3)
coeff_constraint = constraint.fit(
    [torch.tensor(theta, dtype=torch.float32)], [torch.tensor(dt, dtype=torch.float32)]
)[0].numpy()
# %%
error = np.mean(np.abs(coeff_ref - coeff_constraint))
assert error < 1e-5, f"MAE w.r.t reference is too high: {error}"
