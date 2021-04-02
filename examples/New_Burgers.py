#!/usr/bin/env python
# coding: utf-8

# In[23]:


# General imports
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# DeepMoD stuff
from deepymod import DeepMoD
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.constraint import LeastSquares
from deepymod.model.sparse_estimators import Threshold
from deepymod.training import train
from deepymod.training.sparsity_scheduler import TrainTestPeriodic, Periodic, TrainTest

from deepymod.data import Dataset, GPULoader, get_train_test_loader
from deepymod.data.burgers import BurgersDelta

from deepymod.analysis import load_tensorboard

from sklearn.model_selection import train_test_split

# if torch.cuda.is_available():
#     device = "cuda"
# else:
device = "cpu"
print(device)

# In[24]:


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


# In[25]:


dataset = BurgersDelta(
    load_kwargs=load_kwargs, preprocess_kwargs=preprocess_kwargs, device=device
)


# In[ ]:


# In[26]:

# plt.figure()
# plt.scatter(dataset.coords.cpu().numpy()[:, 0], dataset.coords.cpu().numpy()[:, 1])
# plt.show()

# train_idx, test_idx = train_test_split(
#     np.arange(len(dataset)), test_size=0.2, random_state=42
# )
# train_idx = train_idx[:1000]
# test_idx = test_idx[:1000]
# # train_dataloader = DeePyModGPULoader(
# #     dataset, sampler=SubsetRandomSampler(train_idx), batch_size=len(train_idx)
# # )
# # test_dataloader = DeePyModGPULoader(
# #     dataset, sampler=SubsetRandomSampler(test_idx), batch_size=len(test_idx)
# # )
# train_dataloader = DeePyModGPULoader(dataset)
# test_dataloader = DeePyModGPULoader(dataset)

train_dataloader, test_dataloader = get_train_test_loader(
    dataset,
)


# In[27]:


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


# In[ ]:
