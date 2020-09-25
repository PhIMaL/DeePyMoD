# Burgers, tests 1D input

# General imports
import numpy as np
import torch

# DeepMoD stuff
from deepymod_torch.DeepMod import DeepMod
from deepymod_torch.networks import NN

# Setting cuda
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Loading data
data = np.load('data/burgers.npy', allow_pickle=True).item()
X = np.transpose((data['t'].flatten(), data['x'].flatten()))
y = np.real(data['u']).reshape((data['u'].size, 1))
number_of_samples = 1000

idx = np.random.permutation(y.size)
X_train = torch.tensor(X[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)

## Running DeepMoD
config = {'n_in': 2, 'hidden_dims': [20, 20, 20, 20, 20, 20], 'n_out': 1, 'library_function': library_1D_in, 'library_args':{'poly_order': 2, 'diff_order': 2}}

model = DeepMod(**config)
optimizer = torch.optim.Adam([{'params': model.network_parameters(), 'lr':0.002}, {'params': model.coeff_vector(), 'lr':0.002}])
#train_mse(model, X_train, y_train, optimizer, 1000)
train_deepmod(model, X_train, y_train, optimizer, 1000, {'l1': 1e-5})


print()
print(model.fit.sparsity_mask) 
print(model.fit.coeff_vector)