"""
CLASSIFIER TRAINING - SimpleSleepNet
=====================================
Questo file gestisce l'addestramento supervisionato del classificatore.
L'encoder è FREEZ (non viene addestrato), si addestra solo il classificatore
per predire i 5 stadi del sonno (W, N1, N2, N3, REM).

Filosofia:
    1. L'encoder ha già imparato rappresentazioni utili (contrastive learning)
    2. Si aggiunge un classificatore MLP in cima
    3. Si addestra SOLO il classificatore mantenendo l'encoder congelato
    4. Questo è chiamato "linear evaluation" o "fine-tuning"
"""

import torch
import logging
from utils.tensorboard_logger import get_tensorboard_logger
import time
from tqdm import tqdm  # Barra di progresso
import sys

logger = logging.getLogger(__name__)


def evaluate_classifier(encoder, classifier, data_loader, criterion, device='cuda'):
    """
    VALUTA IL CLASSIFICATORE SU UN DATASET
    =======================================
    
    Cosa fa:
        1. Passa i dati attraverso l'encoder (frozen) per ottenere embeddings
        2. Passa gli embeddings attraverso il classificatore
        3. Calcola loss e accuratezza
    
    Differenza dal training:
        - model.eval(): disattiva dropout
        - torch.no_grad(): disabilita gradienti (risparmia memoria)
        - Non aggiorna i pesi
    
    Parametri:
    ----------
    encoder : SimpleSleepNet
        Encoder pre-addestrato (frozen, non si addestra)
    classifier : SleepStageClassifier
        Classificatore MLP da valutare
    data_loader : DataLoader
        Dataset di validazione o test
    criterion : nn.Module
        Funzione di loss (CrossEntropyLoss)
    device : str
        'cuda', 'mps', o 'cpu'
    
    Returns:
    --------
    avg_loss : float
        Loss media sul dataset
    accuracy : float
        Accuratezza (0-1), es. 0.8868 = 88.68%
    """
    try:
        # Modalità valutazione (dropout disattivato)
        encoder.eval()
        classifier.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Disabilita gradienti (risparmia memoria e velocizza)
        with torch.no_grad():
            for inputs, labels in data_loader:
                # Sposta dati sul device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # FORWARD PASS
                embeddings = encoder(inputs)      # encoder produce embeddings
                outputs = classifier(embeddings)  # classificatore produce logits
                loss = criterion(outputs, labels)
                
                # Accumula loss
                total_loss += loss.item()
                
                # Calcola predizioni corrette
                _, predictions = torch.max(outputs, 1)  # classe con probabilità massima
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        # Calcola metriche finali
        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
        
    except Exception as e:
        logger.error("Error during evaluation: %s", str(e))
        raise


def save_model(classifier, save_path):
    """
    SALVA IL CLASSIFICATORE SU DISCO
    =================================
    
    Salva solo i pesi (state_dict), non l'architettura.
    """
    try:
        torch.save(classifier.state_dict(), save_path)
        logger.info("Saved best model to %s", save_path)
    except Exception as e:
        logger.error("Error saving model: %s", str(e))
        raise


def train_epoch(encoder, classifier, train_loader, criterion, optimizer, device, epoch):
    """
    ADDESTRA IL CLASSIFICATORE PER UNA SINGOLA EPOCH
    ================================================
    
    Cosa fa:
        1. Prende batch di (segnale, label)
        2. Passa il segnale attraverso encoder (frozen) → embeddings
        3. Passa embeddings attraverso classificatore → logits
        4. Calcola CrossEntropyLoss tra logits e label vere
        5. Backpropagation e aggiornamento SOLO del classificatore
    
    NOTA: L'encoder è in modalità eval() e i suoi gradienti sono disabilitati
    con torch.no_grad(). Solo i parametri del classificatore vengono aggiornati.
    
    Parametri:
    ----------
    encoder : SimpleSleepNet
        Encoder pre-addestrato (FROZEN)
    classifier : SleepStageClassifier
        Classificatore da addestrare
    train_loader : DataLoader
        Dataset di training con (segnale, label)
    criterion : nn.Module
        Funzione di loss (CrossEntropyLoss)
    optimizer : Optimizer
        Ottimizzatore per il classificatore (es. Adam)
    device : str
        'cuda', 'mps', o 'cpu'
    epoch : int
        Numero epoca corrente (per logging)
    
    Returns:
    --------
    avg_train_loss : float
        Loss media sull'epoca
    epoch_duration : float
        Durata dell'epoca in secondi
    """
    try:
        tensorboard_logger = get_tensorboard_logger()
        
        # Modalità training per il classificatore (dropout attivo)
        classifier.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        # BARRA DI PROGRESSO (VISIBILE NEL TERMINALE)
        # file=sys.stdout forza l'output al terminale
        # leave=True mantiene la barra dopo il completamento
        pbar = tqdm(
            train_loader,
            desc=f'Classifier Epoch {epoch+1}',
            file=sys.stdout,   # <-- FORZA OUTPUT AL TERMINALE
            leave=True         # <-- MANTIENE LA BARRA
        )
        
        for inputs, labels in pbar:
            # Sposta dati sul device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # FORWARD PASS CON ENCODER FREEZ
            # torch.no_grad(): l'encoder non calcola gradienti (risparmia memoria)
            with torch.no_grad():
                embeddings = encoder(inputs)
            
            # Zero gradienti (importante! altrimenti si accumulano)
            optimizer.zero_grad()
            
            # FORWARD PASS DEL CLASSIFICATORE
            outputs = classifier(embeddings)
            
            # CALCOLA LOSS (CrossEntropy = Softmax + Negative Log Likelihood)
            loss = criterion(outputs, labels)
            
            # BACKWARD PASS: calcola gradienti per il classificatore
            loss.backward()
            
            # AGGIORNAMENTO PESI: ottimizzatore modifica i parametri del classificatore
            optimizer.step()
            
            # Accumula loss per la media
            total_loss += loss.item()
            
            # Calcola accuratezza per la barra di progresso
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            current_acc = 100. * correct / total
            
            # Aggiorna la barra di progresso con loss e accuracy correnti
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.2f}%'})
        
        # Calcola metriche finali dell'epoca
        avg_train_loss = total_loss / len(train_loader)
        epoch_duration = time.time() - start_time
        
        # Logging per TensorBoard
        tensorboard_logger.add_scalar('Train/Loss', avg_train_loss, epoch)
        tensorboard_logger.add_scalar('Train/Epoch_Duration', epoch_duration, epoch)
        
        return avg_train_loss, epoch_duration
        
    except Exception as e:
        logger.error("Error during training epoch: %s", str(e))
        raise


def train_classifier(
    encoder,
    classifier,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=50,
    device='cuda',
    save_path='best_classifier/best_classifier_default.pth',
    check_interval=25,
    min_improvement=0.01
):
    """
    LOOP PRINCIPALE DI ADDESTRAMENTO DEL CLASSIFICATORE
    ====================================================
    
    Questa funzione coordina l'addestramento supervisionato del classificatore.
    L'encoder rimane FREEZ (congelato) durante tutto il processo.
    
    Flow:
        1. Per ogni epoca: chiama train_epoch() che addestra SOLO il classificatore
        2. Ogni 'check_interval' epoche: calcola validation loss e accuracy
        3. Se validation loss migliora: salva il modello
        4. Se non migliora per 'check_interval' epoche: early stopping
    
    Perché tenere l'encoder frozen?
        - Preserva le rappresentazioni apprese dal contrastive learning
        - Previene overfitting (pochi dati etichettati)
        - Molto più veloce (meno parametri da addestrare)
    
    Parametri:
    ----------
    encoder : SimpleSleepNet
        Encoder pre-addestrato (FROZEN)
    classifier : SleepStageClassifier
        Classificatore da addestrare
    train_loader : DataLoader
        Dataset di training con etichette
    val_loader : DataLoader
        Dataset di validazione
    criterion : nn.Module
        Funzione di loss (CrossEntropyLoss)
    optimizer : Optimizer
        Ottimizzatore per il classificatore
    num_epochs : int
        Numero massimo di epoche
    device : str
        'cuda', 'mps', o 'cpu'
    save_path : str
        Dove salvare il miglior classificatore
    check_interval : int
        Ogni quante epoche fare validazione
    min_improvement : float
        Miglioramento minimo per considerare progresso
    
    Returns:
    --------
    best_val_loss : float
        Miglior validation loss raggiunto
    """
    try:
        tensorboard_logger = get_tensorboard_logger()
        
        # FREEZA L'ENCODER: non si addestra, solo forward pass
        # Questo è fondamentale! L'encoder rimane come è stato pre-addestrato
        encoder.eval()
        
        # Sposta il classificatore sul device corretto
        classifier.to(device)
        
        # Variabili per early stopping
        best_val_loss = float('inf')
        best_accuracy = 0.0
        total_epochs = 0
        epochs_since_improvement = 0
        
        logger.info("Starting training for %d epochs", num_epochs)
        logger.info("Encoder is FROZEN - only classifier parameters are being updated")
        
        while total_epochs < num_epochs:
            # ---------- ADDESTRA PER check_interval EPOCH ----------
            for _ in range(check_interval):
                if total_epochs >= num_epochs:
                    break
                total_epochs += 1
                
                # Addestra per una epoca
                avg_train_loss, epoch_duration = train_epoch(
                    encoder, classifier, train_loader, criterion, optimizer, device, total_epochs
                )
                logger.info(
                    "Epoch [%d/%d], Train Loss: %.4f, Duration: %.2f sec",
                    total_epochs, num_epochs, avg_train_loss, epoch_duration
                )

            # ---------- VALIDAZIONE ----------
            val_loss, val_accuracy = evaluate_classifier(
                encoder, classifier, val_loader, criterion, device
            )
            logger.info(
                "Validation Loss after %d epochs: %.4f, Validation Accuracy: %.4f",
                total_epochs, val_loss, val_accuracy
            )
            
            # Logging per TensorBoard
            tensorboard_logger.add_scalar('Validation/Loss', val_loss, total_epochs)
            tensorboard_logger.add_scalar('Validation/Accuracy', val_accuracy, total_epochs)
            
            # Verifica se c'è stato un miglioramento
            improvement = best_val_loss - val_loss
            
            if improvement > min_improvement:
                # MIGLIORAMENTO! Salva il modello e resetta il contatore
                best_val_loss = val_loss
                best_accuracy = val_accuracy
                epochs_since_improvement = 0
                save_model(classifier, save_path)
                logger.info("Improved validation loss (Δ = %.4f). Model saved to %s.", improvement, save_path)
                
                # Logging checkpoint per TensorBoard
                tensorboard_logger.add_scalar('Checkpoint/Best_Loss', best_val_loss, total_epochs)
                tensorboard_logger.add_scalar('Checkpoint/Best_Accuracy', best_accuracy, total_epochs)
            else:
                # NESSUN MIGLIORAMENTO
                epochs_since_improvement += check_interval
                logger.info("No significant improvement (Δ = %.4f).", improvement)
                
                # EARLY STOPPING
                if epochs_since_improvement >= check_interval:
                    logger.info("Early stopping due to no significant improvement.")
                    break
        
        # Riepilogo finale
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Best validation accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        logger.info("=" * 60)
        
        return best_val_loss
        
    except Exception as e:
        logger.error("Error during training: %s", str(e))
        raise