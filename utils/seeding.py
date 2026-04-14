import random
import os
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across various libraries.

    Parameters:
    - seed (int): The seed value to be set.
    """
    # Python's built-in random module
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # Ensure deterministic behavior in PyTorch (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash-based operations
    # os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed}")