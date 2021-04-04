#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt

# General imports
import numpy as np
import torch

# DeePyMoD imports
from deepymod import DeepMoD
from deepymod.analysis import load_tensorboard
from deepymod.data import Dataset, GPULoader, get_train_test_loader
from deepymod.data.burgers import burgersdelta
from deepymod.model.constraint import LeastSquares
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold
from deepymod.training import train
from deepymod.training.sparsity_scheduler import Periodic, TrainTest, TrainTestPeriodic
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# if torch.cuda.is_available():
#     device = "cuda"
# else:
device = "cpu"
print(device)

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Making dataset
v = 0.1
A = 1.0

x = torch.linspace(-3, 4, 100)
t = torch.linspace(0.5, 5.0, 50)
load_kwargs = {"x": x, "t": t, "v": v, "A": A}
preprocess_kwargs = {"noise": 0.4}

dataset = Dataset(
    burgersdelta,
    load_kwargs=load_kwargs,
    preprocess_kwargs=preprocess_kwargs,
    device=device,
)

train_dataloader, test_dataloader = get_train_test_loader(
    dataset,
)

network = NN(2, [30, 30, 30, 30, 30], 1)
library = Library1D(poly_order=2, diff_order=3)  # Library function
estimator = Threshold(0.1)  # Sparse estimator
constraint = LeastSquares()  # How to constrain
model = DeepMoD(network, library, estimator, constraint).to(
    device
)  # Putting it all in the model

# sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=200, delta=1e-5) # in terms of write iterations
# sparsity_scheduler = Periodic(initial_iteration=1000, periodicity=25)
sparsity_scheduler = TrainTest(patience=200, delta=1e-5)

optimizer = torch.optim.Adam(
    model.parameters(), betas=(0.99, 0.999), amsgrad=True, lr=2e-3
)  # Defining optimizer

train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    sparsity_scheduler,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=1000,
    delta=0.001,
    patience=200,
)
