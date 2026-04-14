"""
CONTRASTIVE TRAINING - SimpleSleepNet
======================================
Questo file gestisce l'addestramento contrastivo dell'encoder (SimpleSleepNet).
L'obiettivo è imparare rappresentazioni significative dei segnali EEG senza
utilizzare le etichette (self-supervised learning).

Filosofia: due views augmentate dello stesso segnale devono avere embeddings simili,
mentre views di segnali diversi devono avere embeddings diversi.
"""

import torch
from losses import nt_xent_loss  # NT-Xent loss (SimCLR)
import logging
from time import time
from utils.tensorboard_logger import get_tensorboard_logger
from tqdm import tqdm  # Barra di progresso
import sys

# Logger per questo modulo
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, device, temperature, epoch):
    """
    ADDESTRA IL MODELLO PER UNA SINGOLA EPOCH
    ========================================
    
    Cosa fa:
        1. Prende batch di coppie contrastive (x_i, x_j)
        2. Passa entrambi attraverso l'encoder per ottenere embeddings
        3. Calcola la NT-Xent loss tra le due views
        4. Aggiorna i pesi del modello (backpropagation)
    
    Parametri:
    ----------
    model : torch.nn.Module
        L'encoder SimpleSleepNet da addestrare
    dataloader : DataLoader
        Fornisce batch di coppie (x_i, x_j) - due views augmentate
    optimizer : torch.optim.Optimizer
        Ottimizzatore (es. Adam, AdamW)
    device : str
        'cuda', 'mps', o 'cpu' - dove eseguire i calcoli
    temperature : float
        Parametro di temperatura per la NT-Xent loss.
        Valori tipici: 0.07, 0.1, 0.5.
        Più è basso, più le predizioni sono "sharp" (discriminative)
    epoch : int
        Numero dell'epoca corrente (per logging)
    
    Returns:
    --------
    average_loss : float
        Loss media su tutti i batch dell'epoca
    """
    tensorboard_logger = get_tensorboard_logger()
    
    # Modalità training (attiva dropout, batch norm, etc.)
    model.train()
    
    total_loss = 0.0
    start_time = time()
    
    # BARRA DI PROGRESSO (VISIBILE NEL TERMINALE)
    # file=sys.stdout forza l'output al terminale anche con logging
    # leave=True mantiene la barra dopo il completamento
    pbar = tqdm(
        enumerate(dataloader), 
        total=len(dataloader), 
        desc=f'Epoch {epoch+1}',
        file=sys.stdout,      # <-- FORZA OUTPUT AL TERMINALE
        leave=True            # <-- MANTIENE LA BARRA
    )
    
    for batch_idx, (x_i, x_j) in pbar:
        # Sposta i dati sul device (GPU/MPS/CPU)
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        
        # Resetta i gradienti (importante! altrimenti si accumulano)
        optimizer.zero_grad()
        
        # FORWARD PASS: calcola embeddings per entrambe le views
        z_i = model(x_i)  # embedding della prima view
        z_j = model(x_j)  # embedding della seconda view
        
        # Calcola la loss contrastiva NT-Xent
        # Più simili sono z_i e z_j, minore è la loss
        loss = nt_xent_loss(z_i, z_j, temperature)
        
        # BACKWARD PASS: calcola i gradienti
        loss.backward()
        
        # AGGIORNAMENTO PESI: ottimizzatore modifica i parametri
        optimizer.step()
        
        # Accumula la loss per calcolare la media a fine epoca
        total_loss += loss.item()
        
        # Aggiorna la barra di progresso con la loss corrente
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calcola metriche dell'epoca
    epoch_duration = time() - start_time
    average_loss = total_loss / len(dataloader)
    
    # Logging per TensorBoard
    tensorboard_logger.add_scalar('Training Loss', average_loss, epoch)
    tensorboard_logger.add_scalar('Epoch Duration', epoch_duration, epoch)
    
    return average_loss


def save_model(model, save_path):
    """
    SALVA IL MODELLO SU DISCO
    ========================
    
    Salva solo i pesi (state_dict), non l'architettura completa.
    Questo permette di ricaricare il modello in seguito.
    """
    torch.save(model.state_dict(), save_path)
    logger.info("Saved best model to %s", save_path)


def compute_validation_loss(model, dataloader, device, temperature):
    """
    CALCOLA LA LOSS SUL VALIDATION SET
    ===================================
    
    Differenza rispetto al training:
        - model.eval(): disattiva dropout e batch norm
        - torch.no_grad(): disabilita il calcolo dei gradienti (risparmia memoria)
        - Non chiama .backward() o .step()
    
    Usata per monitorare l'overfitting e decidere quando fermare l'addestramento.
    """
    model.eval()  # Modalità valutazione (dropout disattivato)
    total_loss = 0.0
    
    # Disabilita il calcolo dei gradienti (risparmia memoria e velocizza)
    with torch.no_grad():
        for x_i, x_j in dataloader:
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            
            # Forward pass (senza gradienti)
            z_i = model(x_i)
            z_j = model(x_j)
            
            # Calcola loss
            loss = nt_xent_loss(z_i, z_j, temperature)
            total_loss += loss.item()
    
    average_loss = total_loss / len(dataloader)
    
    # Torna a modalità training (per le epoche successive)
    model.train()
    
    return average_loss


def train_contrastive_model(model, dataloader, optimizer, device='cuda', num_epochs=5, 
                            temperature=0.1, val_dataloader=None, check_interval=50, 
                            min_improvement=0.01, best_model_path='best_encoder.pth'):
    """
    LOOP PRINCIPALE DI ADDESTRAMENTO CONTRASTIVO
    =============================================
    
    Questa funzione coordina l'addestramento per multiple epoche,
    gestisce la validazione, l'early stopping e il salvataggio del miglior modello.
    
    Flow:
        1. Per ogni epoca: chiama train_epoch()
        2. Ogni 'check_interval' epoche: calcola validation loss
        3. Se validation loss migliora: salva il modello
        4. Se non migliora per 'check_interval' epoche: early stopping
    
    Parametri:
    ----------
    model : SimpleSleepNet
        L'encoder da addestrare
    dataloader : DataLoader
        Dataset di training con coppie contrastive
    optimizer : Optimizer
        Ottimizzatore (Adam, AdamW, SGD, ...)
    device : str
        'cuda', 'mps', o 'cpu'
    num_epochs : int
        Numero massimo di epoche
    temperature : float
        Parametro temperatura per NT-Xent loss
    val_dataloader : DataLoader, optional
        Dataset di validazione (se None, salta validazione)
    check_interval : int
        Ogni quante epoche fare validazione
    min_improvement : float
        Miglioramento minimo per considerare un progresso significativo
    best_model_path : str
        Dove salvare il miglior modello
    """
    tensorboard_logger = get_tensorboard_logger()
    
    # GESTIONE DEVICE
    # Se richiede CUDA ma non disponibile, passa a CPU
    if not torch.cuda.is_available() and device == 'cuda':
        logger.warning("CUDA is not available. Switching to CPU.")
        device = 'cpu'
    
    # Sposta il modello sul device corretto
    model.to(device)
    model.train()
    
    logger.info(f"Starting contrastive training for {num_epochs} epochs on {device}.")
    
    # Variabili per early stopping
    best_val_loss = float('inf')      # Miglior loss di validazione
    epochs_since_improvement = 0      # Epoche senza miglioramenti
    total_epochs = 0                  # Contatore epoche completate

    try:
        while total_epochs < num_epochs:
            # ---------- TRAIN PER UN'EPOCA ----------
            average_loss = train_epoch(
                model, dataloader, optimizer, device, temperature, total_epochs
            )
            logger.info(f"Epoch [{total_epochs + 1}/{num_epochs}], Training Loss: {average_loss:.4f}")
            total_epochs += 1

            # ---------- VALIDAZIONE (a intervalli) ----------
            if val_dataloader is not None and total_epochs % check_interval == 0:
                val_loss = compute_validation_loss(model, val_dataloader, device, temperature)
                logger.info(f"Validation Loss after {total_epochs} epochs: {val_loss:.4f}")
                tensorboard_logger.add_scalar('Validation Loss', val_loss, total_epochs)

                # Verifica se c'è stato un miglioramento significativo
                improvement = best_val_loss - val_loss
                
                if improvement > min_improvement:
                    # Miglioramento! Salva il modello e resetta il contatore
                    best_val_loss = val_loss
                    epochs_since_improvement = 0
                    save_model(model, best_model_path)
                    logger.info(f"Improved validation loss. Model saved to {best_model_path}.")
                    tensorboard_logger.add_scalar('Best Validation Loss', best_val_loss, total_epochs)
                else:
                    # Nessun miglioramento significativo
                    epochs_since_improvement += check_interval
                    logger.info(f"No significant improvement in validation loss (Δ = {improvement:.4f}).")
                    
                    # EARLY STOPPING: se non migliora da troppo tempo, fermati
                    if epochs_since_improvement >= check_interval:
                        logger.info("Early stopping due to no improvement.")
                        break
                        
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise e

    logger.info("Contrastive training completed.")