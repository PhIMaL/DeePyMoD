#!/usr/bin/env python
# coding: utf-8

# # 2D Advection-Diffusion equation

# in this notebook we provide a simple example of the DeepMoD algorithm and apply it on the 2D advection-diffusion equation.

# In[2]:


# General imports
import numpy as np
import torch
import matplotlib.pylab as plt

# DeepMoD functions

from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.model.func_approx import NN
from deepymod.model.library import Library2D
from deepymod.model.constraint import LeastSquares
from deepymod.model.sparse_estimators import Threshold, PDEFIND
from deepymod.training import train
from deepymod.training.sparsity_scheduler import TrainTestPeriodic
from scipy.io import loadmat

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)

# Configuring GPU or CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)


# ## Prepare the data

# Next, we prepare the dataset.

# In[3]:


def create_data():
    data = loadmat("data/advection_diffusion.mat")
    usol = np.real(data["Expression1"]).astype("float32")
    usol = torch.from_numpy(usol.reshape((51, 51, 61, 4))).float()
    print(usol.shape)
    coords = usol[:, :, :, 0:3].reshape(-1, 3)
    print(coords.shape)
    data = usol[:, :, :, 3].reshape(-1, 1)
    print(data.shape)
    return coords, data


# In[4]:


dataset = Dataset(
    create_data,
    preprocess_kwargs={
        "noise_level": 0.1,
        "normalize_coords": True,
        "normalize_data": True,
    },
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 200},
    device=device,
)


# Next we plot the dataset for three different time-points

# In[5]:


train_dataloader, test_dataloader = get_train_test_loader(dataset, train_test_split=0.5)


# In[7]:


print(len(train_dataloader))


# In[8]:


# ## Configuration of DeepMoD

# Configuration of the function approximator: Here the first argument is the number of input and the last argument the number of output layers.

# In[9]:


network = NN(3, [50, 50, 50, 50], 1)


# Configuration of the library function: We select athe library with a 2D spatial input. Note that that the max differential order has been pre-determined here out of convinience. So, for poly_order 1 the library contains the following 12 terms:
# * [$1, u_x, u_y, u_{xx}, u_{yy}, u_{xy}, u, u u_x, u u_y, u u_{xx}, u u_{yy}, u u_{xy}$]

# In[10]:


library = Library2D(poly_order=1)


# Configuration of the sparsity estimator and sparsity scheduler used. In this case we use the most basic threshold-based Lasso estimator and a scheduler that asseses the validation loss after a given patience. If that value is smaller than 1e-5, the algorithm is converged.

# In[11]:


estimator = Threshold(0.1)
sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=10, delta=1e-5)


# Configuration of the sparsity estimator

# In[12]:


constraint = LeastSquares()
# Configuration of the sparsity scheduler


# Now we instantiate the model and select the optimizer

# In[13]:


model = DeepMoD(network, library, estimator, constraint).to(device)

# Defining optimizer
optimizer = torch.optim.Adam(
    model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3
)


# ## Run DeepMoD

# We can now run DeepMoD using all the options we have set and the training data:
# * The directory where the tensorboard file is written (log_dir)
# * The ratio of train/test set used (split)
# * The maximum number of iterations performed (max_iterations)
# * The absolute change in L1 norm considered converged (delta)
# * The amount of epochs over which the absolute change in L1 norm is calculated (patience)

# In[14]:


train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    sparsity_scheduler,
    log_dir="runs/2DAD/",
    max_iterations=5000,
    delta=1e-4,
    patience=8,
)


# Sparsity masks provide the active and non-active terms in the PDE:

# In[15]:


model.sparsity_masks


# estimatior_coeffs gives the magnitude of the active terms:

# In[16]:


print(model.estimator_coeffs())


# In[ ]:


# In[ ]:
