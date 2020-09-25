from typing import List, NewType
import torch

TensorList = NewType("TensorList", List[torch.Tensor])
