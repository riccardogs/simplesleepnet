import numpy as np
import torch
from torch.utils.data import Dataset

class SupervisedEEGDataset(Dataset):
    def __init__(self, eeg_signals):
        self.data = []
        self.labels = []
        
        if isinstance(eeg_signals, dict):
            try:
                for label in range(5):
                    if label in eeg_signals:
                        signals = eeg_signals[label]
                        if isinstance(signals, np.ndarray):
                            for signal in signals:
                                self.data.append(signal)
                                self.labels.append(label)
                        elif isinstance(signals, list):
                            for signal in signals:
                                self.data.append(signal)
                                self.labels.append(label)
            except:
                for label, signals in eeg_signals.items():
                    if isinstance(signals, np.ndarray):
                        for signal in signals:
                            self.data.append(signal)
                            self.labels.append(label)
                    elif isinstance(signals, list):
                        for signal in signals:
                            self.data.append(signal)
                            self.labels.append(label)
        else:
            raise ValueError(f"Formato non supportato: {type(eeg_signals)}")
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)