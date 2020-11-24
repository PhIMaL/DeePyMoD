""" Defines Tensorlist 
Tensorlist (list[torch.Tensor}): a list of torch Tensors."""

from typing import List, NewType
import torch

TensorList = NewType("TensorList", List[torch.Tensor])
