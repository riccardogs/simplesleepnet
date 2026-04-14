"""
SLEEP STAGE CLASSIFIER - MLP PER CLASSIFICAZIONE
=================================================
Questo file definisce il classificatore MLP che trasforma gli embeddings
prodotti dall'encoder SimpleSleepNet in predizioni degli stadi del sonno.

Architettura:
    Input: embedding dall'encoder (latent_dim = 64 o 128)
    ↓
    Linear(128 → 256) + BatchNorm + Mish + Dropout(0.5)
    ↓
    Linear(256 → 128) + BatchNorm + Mish + Dropout(0.5)
    ↓
    Linear(128 → 5) + (nessuna softmax, perché CrossEntropyLoss la include)
    ↓
    Output: logits per 5 classi (W, N1, N2, N3, REM)

Caratteristiche:
    - MLP a 3 layer (2 nascosti, 1 output)
    - BatchNorm per stabilizzare il training
    - Mish activation (più smooth di ReLU)
    - Dropout per regolarizzazione (previene overfitting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MISH ACTIVATION FUNCTION
# ============================================================================
# Riutilizziamo la stessa Mish definita nell'encoder
# Formula: Mish(x) = x * tanh(softplus(x))
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# ============================================================================
# SLEEP STAGE CLASSIFIER - MLP
# ============================================================================
class SleepStageClassifier(nn.Module):
    """
    Classificatore MLP per gli stadi del sonno.
    
    Questo modello prende gli embeddings prodotti dall'encoder (SimpleSleepNet)
    e li classifica in 5 stadi del sonno:
        0: W   (Wake) - Veglia
        1: N1  (NREM1) - Sonno leggero
        2: N2  (NREM2) - Sonno intermedio
        3: N3  (NREM3) - Sonno profondo (onde lente)
        4: REM (REM)   - Sonno paradossale (movimenti oculari rapidi)
    
    Architettura dettagliata:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Input: (Batch, latent_dim) es. (256, 64) o (256, 128)              │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Linear(latent_dim → 256) - Espande le features                      │
    │ BatchNorm1d(256) - Normalizza per batch                             │
    │ Mish() - Attivazione non lineare                                    │
    │ Dropout(p=0.5) - Regolarizzazione (50% neuroni azzerati)            │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Linear(256 → 128) - Riduce dimensioni                               │
    │ BatchNorm1d(128) - Normalizza per batch                             │
    │ Mish() - Attivazione non lineare                                    │
    │ Dropout(p=0.5) - Regolarizzazione                                   │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Linear(128 → 5) - Proiezione sulle 5 classi                         │
    │ (nessuna softmax - CrossEntropyLoss la applica internamente)        │
    └─────────────────────────────────────────────────────────────────────┘
    
    Parametri:
    ----------
    input_dim : int
        Dimensione dell'embedding in input (deve matchare latent_dim dell'encoder)
        Default: 128
    num_classes : int
        Numero di classi di output (5 stadi del sonno)
        Default: 5
    dropout_probs : float
        Tasso di dropout (probabilità di azzerare un neurone)
        Default: 0.5 (50% di dropout durante training)
        
    Raises:
    -------
    ValueError
        Se dropout_probs non è un float
    """
    
    def __init__(self, input_dim: int = 128, num_classes: int = 5, dropout_probs: float = 0.5):
        super(SleepStageClassifier, self).__init__()
        
        # Validazione del tipo di dropout_probs
        if not isinstance(dropout_probs, float):
            raise ValueError("dropout_probs must be a float.")
        
        logger.info(f"Initializing SleepStageClassifier with input_dim={input_dim}, "
                   f"num_classes={num_classes}, dropout_probs={dropout_probs}")
        
        # ====================================================================
        # CLASSIFIER MLP
        # ====================================================================
        # Un MLP a 3 layer (2 nascosti, 1 output)
        # BatchNorm: normalizza l'input di ogni layer (media=0, var=1)
        # Mish: attivazione non lineare (alternativa a ReLU)
        # Dropout: regolarizzazione (previene overfitting)
        # ====================================================================
        self.classifier = nn.Sequential(
            # ---------- LAYER 1: Espansione ----------
            # input_dim → 256 (espande le features per maggiore capacità)
            nn.Linear(input_dim, 256),
            # BatchNorm: stabilizza il training, permette learning rate più alti
            nn.BatchNorm1d(256),
            # Mish: attivazione smooth (migliore di ReLU per gradienti)
            Mish(),
            # Dropout: disattiva casualmente il 50% dei neuroni durante training
            nn.Dropout(p=dropout_probs),
            
            # ---------- LAYER 2: Compressione ----------
            # 256 → 128 (comprime le features)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            Mish(),
            nn.Dropout(p=dropout_probs),
            
            # ---------- LAYER 3: Output ----------
            # 128 → 5 (logits per le 5 classi)
            # NOTA: Non usiamo softmax qui perché CrossEntropyLoss la applica internamente
            nn.Linear(128, num_classes)
        )
        
        logger.debug("SleepStageClassifier model architecture created.")
    
    # ========================================================================
    # FORWARD PASS
    # ========================================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del classificatore.
        
        Input:
            x : torch.Tensor
                Embedding dall'encoder di shape (Batch, input_dim)
                Esempio: (256, 64) o (256, 128)
        
        Output:
            x : torch.Tensor
                Logits per le 5 classi di shape (Batch, num_classes)
                Esempio: (256, 5)
                
        NOTA: L'output sono logits (valori reali), non probabilità.
        Per ottenere probabilità: torch.softmax(x, dim=1)
        Per ottenere classi predette: torch.argmax(x, dim=1)
        
        Processo:
            1. Primo layer lineare + BatchNorm + Mish + Dropout
            2. Secondo layer lineare + BatchNorm + Mish + Dropout
            3. Layer di output (logits)
        """
        logger.debug("Forward pass started.")
        
        # Passa attraverso tutti i layer del MLP
        x = self.classifier(x)
        
        logger.debug("Forward pass completed.")
        
        # Restituisce logits (non softmax)
        # La loss CrossEntropyLoss applicherà softmax internamente
        return x

""" 
# ============================================================================
# NOTE SULLE DIMENSIONI (con batch_size=256, input_dim=128)
# ============================================================================
#
# Input:                    (256, 128)
#                           ↓
# Linear(128→256):          (256, 256)
# BatchNorm1d(256):         (256, 256)
# Mish():                   (256, 256)
# Dropout(0.5):             (256, 256)  (metà dei neuroni azzerati casualmente)
#                           ↓
# Linear(256→128):          (256, 128)
# BatchNorm1d(128):         (256, 128)
# Mish():                   (256, 128)
# Dropout(0.5):             (256, 128)
#                           ↓
# Linear(128→5):            (256, 5)  ← LOGITS
#
# Per ottenere probabilità: softmax(logits, dim=1) → (256, 5)
# Per ottenere classi:      argmax(logits, dim=1) → (256,)
# ============================================================================


# ============================================================================
# CONFRONTO CON L'ENCODER
# ============================================================================
#
| Aspetto              | Encoder (SimpleSleepNet)     | Classificatore (SleepStageClassifier) |
|---------------------|------------------------------|---------------------------------------|
| **Scopo**           | Estrarre features             | Classificare stadi del sonno          |
| **Input**           | Segnale EEG grezzo (3000)     | Embedding (64 o 128)                  |
| **Output**          | Embedding normalizzato        | Logits per 5 classi                   |
| **Addestrato con**  | Contrastive learning (senza labels) | Supervised (con labels)          |
| **Dropout**         | 0.2                           | 0.5                                   |
| **BatchNorm**       | Sì                            | Sì                                    |
| **Attivazione**     | Mish                          | Mish                                  |
# ============================================================================


# ============================================================================
# PERCHÉ QUESTO CLASSIFICATORE?
# ============================================================================
#
| Scelta              | Motivazione                                           |
|--------------------|-------------------------------------------------------|
| **MLP a 3 layer**  | Abbastanza espressivo per separare 5 classi           |
| **BatchNorm**      | Stabilizza training, permette learning rate più alti  |
| **Dropout(0.5)**   | Forte regolarizzazione (essenziale con pochi dati)    |
| **Mish**           | Più smooth di ReLU, gradienti migliori               |
| **Nessuna softmax**| CrossEntropyLoss la applica internamente (stabile)   |
# ============================================================================

"""