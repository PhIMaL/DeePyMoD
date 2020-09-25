# Diffusion, tests 2D input, D = 0.5

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
data = np.load('data/diffusion_2D.npy', allow_pickle=True).item()
X = np.transpose((data['t'].flatten(), data['x'].flatten(), data['y'].flatten()))
y = np.real(data['u']).reshape((data['u'].size, 1))
number_of_samples = 1000

idx = np.random.permutation(y.size)
X_train = torch.tensor(X[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y[idx, :][:number_of_samples], dtype=torch.float32)

## Running DeepMoD
config = {'input_dim': 3, 'hidden_dim': 20, 'layers': 5, 'output_dim': 1, 'library_function': library_basic, 'library_args':{'poly_order': 1, 'diff_order': 2}}

X_input = create_deriv_data(X_train, config['library_args']['diff_order'])

model = DeepMod(config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
model.train(X_input, y_train, optimizer, 5000, type='deepmod')

print()
print(model.sparsity_mask_list) 
print(model.coeff_vector_list)