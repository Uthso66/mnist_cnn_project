# src/utils.py
import random
import numpy as np
import torch

def set_seed(seed=0):
    """
    Set seed for reproducibility across random, numpy, and torch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
