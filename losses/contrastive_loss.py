"""
CONTRASTIVE LOSS - NT-Xent (Normalized Temperature-scaled Cross Entropy Loss)

Questa è la loss function introdotta da SimCLR (Google Research, 2020).
L'obiettivo è avvicinare gli embeddings di due views AUGMENTATE dello stesso segnale,
e allontanare gli embeddings di segnali diversi.

Concetti chiave:
    - Positive pairs: due views augmentate dello STESSO segnale
    - Negative pairs: views di segnali DIVERSI
    - Temperature: controlla la "sharpness" delle similarità
    - InfoNCE: massimizza l'information mutual tra positive pairs

Formula matematica semplificata:
    Loss = -log( exp(sim(z_i, z_j)/τ) / Σ exp(sim(z_i, z_k)/τ) )
    dove sim = similarità coseno, τ = temperatura
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def normalize_embeddings(embedding_1, embedding_2):
    """
    NORMALIZZA GLI EMBEDDINGS LUNGO LA DIMENSIONE DELLE FEATURES
    ============================================================
    
    Cosa fa:
        Applica L2-normalization agli embeddings.
        Dopo la normalizzazione, il prodotto scalare = similarità coseno.
    
    Perché è importante:
        - La similarità coseno è compresa tra -1 e 1
        - È indipendente dalla magnitudine degli embeddings
        - Previene il collapse delle rappresentazioni
    
    Formula:
        z_normalized = z / ||z||_2
        
    Parametri:
    ----------
    embedding_1, embedding_2 : torch.Tensor
        Shape: (batch_size, embedding_dim)
        
    Returns:
    --------
    embedding_1, embedding_2 : torch.Tensor
        Embeddings normalizzati (stessa shape)
    """
    # L2-normalization lungo la dimensione 1 (features)
    # p=2 = norma euclidea
    embedding_1 = nn.functional.normalize(embedding_1, dim=1, p=2)
    embedding_2 = nn.functional.normalize(embedding_2, dim=1, p=2)
    return embedding_1, embedding_2


def concatenate_embeddings(embedding_1, embedding_2):
    """
    CONCATENA GLI EMBEDDINGS LUNGO LA DIMENSIONE DEL BATCH
    ======================================================
    
    Cosa fa:
        Prende due batch di embeddings (ciascuno di size B x D) e
        li concatena in un unico tensore di size (2B x D).
    
    Perché:
        Per calcolare la matrice di similarità tra TUTTE le coppie
        in un unico passaggio efficiente.
    
    Esempio:
        batch_size = 3, embedding_dim = 128
        embedding_1 shape: (3, 128)
        embedding_2 shape: (3, 128)
        output shape: (6, 128)
    
    Returns:
    --------
    embeddings : torch.Tensor
        Shape: (2*batch_size, embedding_dim)
    """
    return torch.cat([embedding_1, embedding_2], dim=0)


def compute_similarity_matrix(embeddings, temperature):
    """
    CALCOLA LA MATRICE DI SIMILARITÀ COSENO
    =======================================
    
    Cosa fa:
        Calcola la similarità tra OGNI coppia di embeddings.
        Poiché gli embeddings sono normalizzati, il prodotto scalare
        è equivalente alla similarità coseno.
        
    Formula:
        similarity_matrix[i, j] = (embeddings[i] · embeddings[j]) / temperature
    
    Parametri:
    ----------
    embeddings : torch.Tensor
        Shape: (2*batch_size, embedding_dim) - embeddings concatenati
    temperature : float
        Parametro di temperatura. Più è basso, più le similarità sono "sharp"
        (valori estremi). Tipicamente 0.07, 0.1, o 0.5.
    
    Returns:
    --------
    similarity_matrix : torch.Tensor
        Shape: (2*batch_size, 2*batch_size)
        Ogni cella [i,j] contiene la similarità tra embedding i e embedding j
    """
    # Prodotto scalare tra tutti gli embeddings: E * E^T
    # Poiché gli embeddings sono normalizzati, questo = similarità coseno
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    
    # Scala per temperatura (più temperatura bassa = più discriminativo)
    similarity_matrix = similarity_matrix / temperature
    
    return similarity_matrix


def mask_self_similarities(similarity_matrix, batch_size, device):
    """
    MASCHERA LE SIMILARITÀ DI UN VETTORE CON SE STESSO
    ==================================================
    
    Cosa fa:
        Mette a -infinito le celle della diagonale (similarità di un embedding con se stesso).
        In questo modo, durante il softmax, queste celle avranno probabilità 0.
    
    Perché:
        Non vogliamo che il modello impari a predire che un embedding è simile
        a se stesso (banale). Vogliamo solo coppie positive (i, i+batch_size)
        e coppie negative (tutte le altre).
    
    Esempio con batch_size=2:
        similarity_matrix (4x4):
            [i0,i0] [i0,i1] [i0,j0] [i0,j1]
            [i1,i0] [i1,i1] [i1,j0] [i1,j1]
            [j0,i0] [j0,i1] [j0,j0] [j0,j1]
            [j1,i0] [j1,i1] [j1,j0] [j1,j1]
        
        Dopo masking:
            [-inf,  ...]  (diagonale principale a -inf)
    
    Parametri:
    ----------
    similarity_matrix : torch.Tensor
        Shape: (2*batch_size, 2*batch_size)
    batch_size : int
        Dimensione del batch originale
    device : torch.device
        'cuda', 'mps', o 'cpu'
    
    Returns:
    --------
    similarity_matrix : torch.Tensor
        Matrice con la diagonale mascherata a -inf
    """
    # Crea matrice identità (1 sulla diagonale, 0 altrove)
    # Shape: (2*batch_size, 2*batch_size)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
    
    # Metti a -infinito le posizioni dove mask è True (la diagonale)
    # -inf fa sì che il softmax dia probabilità 0 a queste posizioni
    similarity_matrix.masked_fill_(mask, -float('inf'))
    
    return similarity_matrix


def nt_xent_loss(embedding_1, embedding_2, temperature=0.5):
    """
    NT-XENT LOSS (Normalized Temperature-scaled Cross Entropy Loss)
    ===============================================================
    
    Questa è la loss function principale del contrastive learning (SimCLR).
    
    IDEA INTUITIVA:
        - Abbiamo N segnali originali
        - Da ogni segnale generiamo 2 views augmentate → 2N embeddings totali
        - Per ogni anchor (un embedding), vogliamo che:
            * Il suo embedding "gemello" (l'altra view dello stesso segnale)
              sia quello con similarità massima
            * Tutti gli altri 2N-2 embeddings (views di segnali diversi)
              abbiano similarità bassa
    
    FORMULA:
        Loss = -log( exp(sim(z_i, z_j)/τ) / Σ_{k=1}^{2N} exp(sim(z_i, z_k)/τ) )
        dove:
            - z_i, z_j sono le due views dello stesso segnale (positive pair)
            - z_k sono tutti gli embeddings (positivi + negativi)
            - τ (tau) è la temperatura
            - sim è la similarità coseno
    
    PARAMETRI IMPORTANTI:
        - temperature (τ): 
            * Bassa (es. 0.07) → decision boundary più sharp, embeddings più discriminativi
            * Alta (es. 0.5) → learning più smooth, più stabile
        - batch_size: 
            * Più grande → più coppie negative → rappresentazioni migliori
    
    FLUSSO DEI DATI:
        Input: 
            embedding_1 shape (B, D) - prima view augmentata
            embedding_2 shape (B, D) - seconda view augmentata
        
        Dopo normalizzazione e concatenazione:
            embeddings shape (2B, D)
        
        Matrice di similarità:
            S shape (2B, 2B) dove S[i,j] = cos(emb[i], emb[j]) / τ
        
        Dopo masking diagonale:
            S[i,i] = -inf (escludiamo similarità con se stesso)
        
        Labels:
            Per i = 0..B-1:   positive = i+B (l'embedding gemello)
            Per i = B..2B-1: positive = i-B (l'embedding gemello)
        
        Loss = CrossEntropy(S, labels)
    
    Parametri:
    ----------
    embedding_1 : torch.Tensor
        Embeddings della prima view augmentata (batch_size, embedding_dim)
    embedding_2 : torch.Tensor
        Embeddings della seconda view augmentata (batch_size, embedding_dim)
    temperature : float
        Parametro di temperatura. Default: 0.5 (come in SimCLR originale)
        Valori tipici: 0.07, 0.1, 0.5
    
    Returns:
    --------
    loss : torch.Tensor
        Scalare: la loss contrastiva per il batch
    
    ESEMPIO CON BATCH_SIZE=2:
        Segnali originali: A, B
        Views: A1, A2, B1, B2
        
        Vogliamo che:
            A1 sia simile ad A2 (e non a B1, B2)
            A2 sia simile ad A1 (e non a B1, B2)
            B1 sia simile a B2 (e non a A1, A2)
            B2 sia simile a B1 (e non a A1, A2)
    """
    batch_size = embedding_1.size(0)
    device = embedding_1.device

    # STEP 1: Normalizza embeddings (similarità coseno)
    embedding_1, embedding_2 = normalize_embeddings(embedding_1, embedding_2)
    
    # STEP 2: Concatena per calcolare tutte le similarità in una volta
    embeddings = concatenate_embeddings(embedding_1, embedding_2)
    
    # STEP 3: Calcola matrice di similarità (2B x 2B)
    similarity_matrix = compute_similarity_matrix(embeddings, temperature)
    
    # STEP 4: Maschera la diagonale (similarità con se stesso)
    similarity_matrix = mask_self_similarities(similarity_matrix, batch_size, device)

    # STEP 5: Crea le etichette per le positive pairs
    # Per i primi B embeddings (view1), il positivo è l'embedding i+B (view2)
    # Per gli ultimi B embeddings (view2), il positivo è l'embedding i-B (view1)
    labels = torch.arange(batch_size).to(device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    
    # STEP 6: Calcola CrossEntropyLoss
    # CrossEntropyLoss fa automaticamente softmax + log + NLL
    loss = nn.CrossEntropyLoss()(similarity_matrix, labels)

    return loss