"""
SIMPLE SLEEP NET - ENCODER CNN 1D
==================================
Questo file definisce l'architettura dell'encoder che trasforma i segnali EEG
in embeddings vettoriali. Questi embeddings vengono poi usati:
    - Nel contrastive learning: per calcolare la similarità tra views
    - Nel classificatore: come features per predire gli stadi del sonno

Architettura:
    Input: segnale EEG grezzo (1 canale, 3000 campioni)
    ↓
    CNN 1D a 3 blocchi (convoluzioni dilate)
    ↓
    Global Average Pooling
    ↓
    MLP (Linear + BatchNorm + Mish)
    ↓
    Output: embedding normalizzato L2 (latent_dim dimensioni)

Caratteristiche:
    - Convoluzioni dilate: aumentano il receptive field senza perdere risoluzione
    - Mish activation: più smooth di ReLU, migliori gradienti
    - L2 normalization: embeddings su sfera unitaria (similarità coseno)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MISH ACTIVATION FUNCTION
# ============================================================================
# La Mish è una funzione di attivazione proposta nel paper "Mish: A Self Regularized
# Non-Monotonic Activation Function" (Diganta Misra, 2019).
#
# Formula: Mish(x) = x * tanh(softplus(x))
# dove softplus(x) = log(1 + e^x)
#
# Vantaggi rispetto a ReLU:
#   1. Non monotona (permette gradienti negativi piccoli)
#   2. Smooth (derivata continua, miglior training)
#   3. Self-regularizing (leggermente)
#   4. Spesso outperforma ReLU su task complessi
#
# Grafico: forma a "s" morbida, valori negativi piccoli vicino a zero
# ============================================================================
class Mish(nn.Module):
    """
    Mish Activation Function: x * tanh(softplus(x))
    
    Softplus(x) = log(1 + e^x) è una approssimazione smooth di ReLU.
    """
    def forward(self, x):
        # F.softplus(x) = log(1 + exp(x))
        # torch.tanh() = (e^x - e^{-x}) / (e^x + e^{-x})
        return x * torch.tanh(F.softplus(x))


# ============================================================================
# SIMPLE SLEEP NET - ENCODER
# ============================================================================
class SimpleSleepNet(nn.Module):
    """
    SimpleSleepNet: Encoder CNN 1D per segnali EEG
    
    Architettura dettagliata:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Input: (Batch, 1 canale, 3000 campioni)                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Conv1d: 1→32 canali, kernel=64, stride=8, padding=63, dilation=1   │
    │ BatchNorm1d(32) + Mish() + Dropout(0.2)                            │
    │   → Dopo conv: (Batch, 32, ~375)                                    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Conv1d: 32→64 canali, kernel=32, stride=4, padding=62, dilation=2  │
    │ BatchNorm1d(64) + Mish() + Dropout(0.2)                            │
    │   → Dopo conv: (Batch, 64, ~94)                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Conv1d: 64→128 canali, kernel=16, stride=2, padding=60, dilation=4 │
    │ BatchNorm1d(128) + Mish() + Dropout(0.2)                           │
    │   → Dopo conv: (Batch, 128, ~47)                                    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ AdaptiveAvgPool1d(1) → (Batch, 128, 1)                             │
    │ Flatten → (Batch, 128)                                              │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Linear(128 → latent_dim)                                            │
    │ BatchNorm1d(latent_dim) + Mish() + Dropout(0.2)                    │
    │   → Output: (Batch, latent_dim)                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │ L2 Normalization (proietta embeddings sulla sfera unitaria)        │
    └─────────────────────────────────────────────────────────────────────┘
    
    Perché convoluzioni dilate?
        - Dilated convolution: salta 'dilation-1' campioni tra input e kernel
        - Aumenta il receptive field senza aumentare il numero di parametri
        - Permette di catturare pattern a diverse scale temporali
        
    Parametri:
    ----------
    latent_dim : int
        Dimensione dello spazio latente (embedding finale). Default: 128
    dropout : float
        Tasso di dropout (probabilità di azzerare un neurone). Default: 0.2
    """
    
    def __init__(self, latent_dim=128, dropout=0.2):
        super(SimpleSleepNet, self).__init__()
        
        logger.info(f"Initializing SimpleSleepNet with latent_dim={latent_dim} and dropout={dropout}")
        
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(p=dropout)
        
        # ====================================================================
        # CONVOLUTIONAL PATH (Feature Extractor)
        # ====================================================================
        # 3 blocchi convolutivi con:
        #   - Conv1d: estrae pattern temporali
        #   - BatchNorm: normalizza per stabilizzare il training
        #   - Mish: attivazione non lineare
        #   - Dropout: regolarizzazione (previene overfitting)
        # ====================================================================
        self.conv_path = nn.Sequential(
            # ---------- BLOCCO 1 ----------
            # Input: (B, 1, 3000)
            # Output: (B, 32, ~375)
            nn.Conv1d(
                in_channels=1,      # EEG è mono-canale (Fpz-Cz)
                out_channels=32,    # Numero di filtri (feature maps)
                kernel_size=64,     # Grandezza del filtro (cattura ~64 campioni)
                stride=8,           # Quanto spostare il filtro (riduce dimensione)
                padding=63,         # Mantiene la dimensione? No, stride la riduce
                dilation=1,         # Dilation base (nessun salto)
                bias=False          # BatchNorm già ha un bias
            ),
            nn.BatchNorm1d(32),     # Normalizza per canale (media=0, var=1)
            Mish(),                  # Attivazione non lineare
            self.dropout,           # Regolarizzazione
            
            # ---------- BLOCCO 2 ----------
            # Input: (B, 32, ~375)
            # Output: (B, 64, ~94)
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=32,
                stride=4,
                padding=62,
                dilation=2,         # Salta 1 campione tra input (receptive field più largo)
                bias=False
            ),
            nn.BatchNorm1d(64),
            Mish(),
            self.dropout,
            
            # ---------- BLOCCO 3 ----------
            # Input: (B, 64, ~94)
            # Output: (B, 128, ~47)
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=16,
                stride=2,
                padding=60,
                dilation=4,         # Salta 3 campioni tra input (receptive field molto largo)
                bias=False
            ),
            nn.BatchNorm1d(128),
            Mish(),
            self.dropout,
        )
        
        # ====================================================================
        # FULLY CONNECTED PATH (Projection Head)
        # ====================================================================
        # Riduce la dimensione da 128 a latent_dim (es. 128 → 64 o 128)
        # Questo strato è talvolta chiamato "projection head" nel contrastive learning
        # ====================================================================
        self.fc = nn.Sequential(
            nn.Linear(128, self.latent_dim),        # Proiezione lineare
            nn.BatchNorm1d(self.latent_dim),        # Normalizzazione
            Mish(),                                 # Attivazione
            self.dropout                            # Regolarizzazione
        )
        
        logger.info("SimpleSleepNet initialization complete.")
    
    # ========================================================================
    # FORWARD PASS
    # ========================================================================
    def forward(self, x):
        """
        Forward pass del modello.
        
        Input:
            x : torch.Tensor
                Segnale EEG grezzo di shape (Batch, Channels, Length)
                Esempio: (256, 1, 3000) dove 3000 campioni ~30 secondi a 100Hz
        
        Output:
            x : torch.Tensor
                Embedding normalizzato L2 di shape (Batch, latent_dim)
                Esempio: (256, 128) o (256, 64)
                
        Processo:
            1. Estrae features con la CNN 1D
            2. Global average pooling: media su tutta la dimensione temporale
            3. Flatten: (B, C, 1) → (B, C)
            4. Proietta nello spazio latente con MLP
            5. L2 normalization: ||output|| = 1 (proiezione sulla sfera unitaria)
        """
        logger.debug("Starting forward pass.")
        
        # STEP 1: Estrazione features con CNN
        # Input: (B, 1, 3000) → Output: (B, 128, ~47)
        x = self.conv_path(x)
        
        # STEP 2: Global Average Pooling
        # Prende la media su tutta la dimensione temporale (ultima dimensione)
        # Output: (B, 128, 1)
        x = F.adaptive_avg_pool1d(x, 1)
        
        # STEP 3: Flatten
        # Rimuove la dimensione singleton (1) → (B, 128)
        x = x.view(x.size(0), -1)
        
        # STEP 4: Proiezione nello spazio latente
        # (B, 128) → (B, latent_dim)
        x = self.fc(x)
        
        # STEP 5: L2 Normalization
        # Proietta gli embeddings sulla sfera unitaria
        # Questo è fondamentale per il contrastive learning:
        #   - La similarità coseno = prodotto scalare dopo L2-norm
        #   - Tutti gli embeddings hanno la stessa magnitudine
        #   - Previene il collapse delle rappresentazioni
        x = F.normalize(x, p=2, dim=1)
        
        logger.debug("Forward pass complete.")
        return x


# ============================================================================
# NOTE SULLE DIMENSIONI (con batch_size=256, latent_dim=128)
# ============================================================================
#
# Input:           (256, 1, 3000)
#                  
# Dopo Conv1 (64, stride8, pad63, dil1):
#   L_out = floor((3000 + 2*63 - 64*(1) - (64-1)*(1-1)) / 8) + 1
#         = floor((3000 + 126 - 64) / 8) + 1
#         = floor(3062 / 8) + 1 = 382 + 1 = 383? (circa)
#   Output: (256, 32, ~375)
#
# Dopo Conv2 (32→64, k32, s4, pad62, dil2):
#   Output: (256, 64, ~94)
#
# Dopo Conv3 (64→128, k16, s2, pad60, dil4):
#   Output: (256, 128, ~47)
#
# Dopo AdaptiveAvgPool1d(1):
#   Output: (256, 128, 1)
#
# Dopo Flatten:
#   Output: (256, 128)
#
# Dopo Linear(128→128):
#   Output: (256, 128)
#
# Dopo L2 Normalization:
#   Output: (256, 128) con ||output|| = 1 per ogni riga
# ============================================================================