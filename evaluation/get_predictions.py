import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
from torch import nn
from typing import Tuple

logger = logging.getLogger(__name__)

def get_predictions(encoder: nn.Module, classifier: nn.Module, data_loader: DataLoader, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    classifier.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            embeddings = encoder(inputs)
            outputs = classifier(embeddings)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_true_labels)