import random
import numpy as np
import torch
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class ContrastiveEEGDataset(Dataset):
    """
    A PyTorch Dataset for contrastive learning with EEG signals.
    
    This dataset applies augmentations to EEG signals and returns pairs of augmented signals.
    """
    def __init__(self, eeg_signals, augmentations=None, return_labels=False):
        """
        Initialize the ContrastiveEEGDataset.
        
        Parameters:
        eeg_signals (dict): Dictionary containing EEG signals per class.
        augmentations (list): A list of augmentation callables.
        return_labels (bool): Whether to return labels in __getitem__.
        """
        logger.debug(f"Creating ContrastiveEEGDataset with {len(eeg_signals)} classes")
        self.data = np.array([signal for signals in eeg_signals.values() for signal in signals])
        self.labels = np.array([label for label, signals in eeg_signals.items() for _ in signals])
        self.augmentations = augmentations or []
        self.return_labels = return_labels
        self.requires_x_random = any(getattr(aug, 'requires_x_random', False) for aug in self.augmentations)
        logger.info(f"ContrastiveEEGDataset created with {len(self.data)} samples")
        
    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
        int: Number of samples.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a pair of augmented EEG signals and optionally the label.
        
        Parameters:
        idx (int): Index of the sample to retrieve.
        
        Returns:
        tuple: A tuple containing two augmented EEG signals and optionally the label.
        """
        try:
            signal = self.data[idx]
            label = self.labels[idx]
            random_signal = self.data[random.randint(0, len(self.data) - 1)] if self.requires_x_random else None
            augmented_signal_i = self.apply_augmentations(signal, random_signal)
            augmented_signal_j = self.apply_augmentations(signal, random_signal)
            
            # Convert to torch tensors and add channel dimension
            augmented_signal_i = torch.tensor(augmented_signal_i, dtype=torch.float32).unsqueeze(0)
            augmented_signal_j = torch.tensor(augmented_signal_j, dtype=torch.float32).unsqueeze(0)
            
            if self.return_labels:
                label = torch.tensor(label, dtype=torch.long)
                return augmented_signal_i, augmented_signal_j, label
            else:
                return augmented_signal_i, augmented_signal_j
        except Exception as e:
            logger.error(f"Error in __getitem__ at index {idx}: {e}")
            raise

    def apply_augmentations(self, signal, random_signal):
        """
        Apply augmentations to the given EEG signal.
        
        Parameters:
        signal (np.ndarray): EEG signal to augment.
        random_signal (np.ndarray): Random EEG signal for augmentations that require it.
        
        Returns:
        np.ndarray: Augmented EEG signal.
        """
        for aug in self.augmentations:
            signal = aug(signal, random_signal) if getattr(aug, 'requires_x_random', False) else aug(signal)
        return signal