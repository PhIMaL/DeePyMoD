# General imports
import numpy as np
import torch

# DeepMoD stuff
from deepymod import DeepMoD
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.constraint import LeastSquares
from deepymod.model.sparse_estimators import Threshold
from deepymod.training import train
from deepymod.training.sparsity_scheduler import TrainTestPeriodic, Periodic, TrainTest

from deepymod.data import Dataset
from deepymod.data.burgers import BurgersDelta

from deepymod.analysis import load_tensorboard

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Making dataset
v = 0.1
A = 1.0

x = np.linspace(-3, 4, 100)
t = np.linspace(0.5, 5.0, 50)
x_grid, t_grid = np.meshgrid(x, t, indexing='ij')
dataset = Dataset(BurgersDelta, v=v, A=A)
X, y = dataset.create_dataset(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1), n_samples=1000, noise=0.4, random=True, normalize=False)
X, y = X.to(device), y.to(device)
        
network = NN(2, [30, 30, 30, 30, 30], 1)
library = Library1D(poly_order=2, diff_order=3) # Library function
estimator = Threshold(0.1) # Sparse estimator 
constraint = LeastSquares() # How to constrain
model = DeepMoD(network, library, estimator, constraint).to(device) # Putting it all in the model

#sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=200, delta=1e-5) # in terms of write iterations
#sparsity_scheduler = Periodic(initial_iteration=1000, periodicity=25)
sparsity_scheduler = TrainTest(patience=200, delta=1e-5)

optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.999), amsgrad=True, lr=2e-3) # Defining optimizer

train(model, X, y, optimizer, sparsity_scheduler, exp_ID='Test', split=0.8, write_iterations=25, max_iterations=20000, delta=0.001, patience=200) 