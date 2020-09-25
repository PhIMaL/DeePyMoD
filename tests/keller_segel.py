# Keller Segel, tests coupled output.

# General imports
import numpy as np
import torch

# DeepMoD stuff
from deepymod_torch.DeepMod import DeepMod
from deepymod_torch.library_functions import library_basic
from deepymod_torch.utilities import create_deriv_data

# Setting cuda
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Loading data
data = np.load('data/keller_segel.npy', allow_pickle=True).item()
X = np.transpose((data['t'].flatten(), data['x'].flatten()))
y = np.transpose((data['u'].flatten(), data['v'].flatten()))
number_of_samples = 5000

idx = np.random.permutation(y.shape[0])
X_train = torch.tensor(X[idx, :][:number_of_samples], dtype=torch.float32)
y_train = torch.tensor(y[idx, :][:number_of_samples], dtype=torch.float32)

## Running DeepMoD
config = {'input_dim': 2, 'hidden_dim': 20, 'layers': 5, 'output_dim': 2, 'library_function': library_basic, 'library_args':{'poly_order': 1, 'diff_order': 2}}

X_input = create_deriv_data(X_train, config['library_args']['diff_order'])
optimizer = torch.optim.Adam(model.parameters())
model.train(X_input, y_train, optimizer, 100000, type='deepmod')

print()
print(model.sparsity_mask_list) 
print(model.coeff_vector_list)