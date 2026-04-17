import os
import numpy as np
import glob
import logging
from typing import Dict, Optional
import random

logger = logging.getLogger(__name__)


def load_eeg_data(dataset_path: str, num_files_to_process: Optional[int] = None) -> Dict[str, Dict[int, np.ndarray]]:
    """
    CARICA E ORGANIZZA I DATI EEG DA FILE .NPZ
    
    Cosa fa:
        1. Carica i file .npz dalla directory specificata
        2. Divide i soggetti in train (85%) e test (15%)
        3. Organizza i dati in un dizionario per set e per label
    
    Perché lo split per soggetti?
        - Evita che epoch dello stesso soggetto finiscano sia in train che in test
        - Questo è chiamato "subject-wise split" ed è fondamentale per:
            * Evitare data leakage (il modello potrebbe imparare pattern specifici del soggetto)
            * Valutare la generalizzazione su nuovi soggetti
            * Simulare uno scenario reale (nuovo paziente)
    
    Struttura dell'output:
        eeg_data = {
            'train': {
                0: np.array([...]),  # W (Wake) - tutti gli epoch di train
                1: np.array([...]),  # N1 (NREM1)
                2: np.array([...]),  # N2 (NREM2)
                3: np.array([...]),  # N3 (NREM3 - sonno profondo)
                4: np.array([...])   # REM
            },
            'test': {
                0: np.array([...]),  # stesse label per il test
                1: np.array([...]),
                2: np.array([...]),
                3: np.array([...]),
                4: np.array([...])
            }
        }
    
    Parametri:
    ----------
    dataset_path : str
        Percorso alla directory contenente i file .npz
        Esempio: "./dset/Sleep-EDF-2018/npz/Fpz-Cz"
    
    num_files_to_process : int, optional
        Numero di file da processare (utile per test rapidi)
        Se None, processa tutti i file
    
    Returns:
    --------
    eeg_data : dict
        Dizionario con struttura {set: {label: np.array}}
    """
    

    # Inizializza la struttura dati
    # 5 classi: 0=W, 1=N1, 2=N2, 3=N3, 4=REM
    eeg_data = {
        'train': {label: [] for label in range(5)},
        'test': {label: [] for label in range(5)},
    }

    try:

        # STEP 1: TROVA TUTTI I FILE .NPZ
  
        # glob.glob restituisce lista di path che matchano il pattern
        # sorted garantisce ordine consistente tra esecuzioni
        npz_files = sorted(glob.glob(os.path.join(dataset_path, '*.npz')))
        
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {dataset_path}.")
        
        # Limita il numero di file se richiesto (utile per debugging)
        if num_files_to_process is not None:
            npz_files = npz_files[:num_files_to_process]
        
        logger.info(f"Processing {len(npz_files)} npz files from {dataset_path}.")




        # STEP 2: ESTRAI GLI INDICI DEI SOGGETTI

        # I nomi dei file sono tipo "SC4001.npz"
        # I primi 2 caratteri sono "SC", i successivi 2 sono il numero soggetto
        # Esempio: SC4001.npz → subject_idx = 40
      
        subject_indices = []
        for npz_file in npz_files:
            basename = os.path.basename(npz_file)
            # Prende i caratteri 3 e 4 (indice 2:4 in Python)
            # "SC4001" → [2:4] = "40" → int = 40
            subject_idx = int(basename[3:5])
            subject_indices.append(subject_idx)
        
        # Set per ottenere soggetti unici (ogni soggetto ha più file)
        unique_subject_indices = list(set(subject_indices))
        
        logger.info(f"Found {len(unique_subject_indices)} unique subjects.")


  
        # STEP 3: SHUFFLE E SPLIT DEI SOGGETTI
  
      
        # Shuffle per randomizzare la suddivisione
        random.shuffle(unique_subject_indices)
        
        # Calcola dimensione del train (85% dei soggetti)
        total_subjects = len(unique_subject_indices)
        train_size = int(total_subjects * 0.85)      # 85% per training
        test_size = total_subjects - train_size      # 15% per test
        
        logger.info(f"Subjects split: {train_size} train, {test_size} test")

        # Partition: primi train_size soggetti → train, i restanti → test
        train_subjects = unique_subject_indices[:train_size]
        test_subjects = unique_subject_indices[train_size:]

        logger.info(f"Train subjects: {sorted(train_subjects)}")
        logger.info(f"Test subjects: {sorted(test_subjects)}")



        # STEP 4: PROCESSA OGNI FILE E ASSEGNA AL SET CORRETTO

        processed_files = 0
        
        for idx, npz_file in enumerate(npz_files, 1):
            try:
                basename = os.path.basename(npz_file)
                subject_idx = int(basename[3:5])

                # Determina se questo soggetto va in train o test
                if subject_idx in train_subjects:
                    set_name = 'train'
                elif subject_idx in test_subjects:
                    set_name = 'test'
                else:
                    # Non dovrebbe succedere, ma per sicurezza
                    logger.warning(f"Subject {subject_idx} not in any set")
                    continue

                # Carica il file .npz
                with np.load(npz_file) as data:
                    eeg_epochs = data['x']   # Shape: (N_epochs, 1, 3000)
                    labels = data['y']        # Shape: (N_epochs,)
                    
                    # Separa gli epoch per label (0-4)
                    # Questo permette un accesso più facile in seguito
                    for label in range(5):
                        # Maschera booleana: True dove la label è quella cercata
                        mask = labels == label
                        if np.any(mask):  # Se ci sono epoch con questa label
                            # Aggiunge tutti gli epoch di questa label alla lista
                            eeg_data[set_name][label].extend(eeg_epochs[mask])
                
                processed_files += 1
                
                # Log ogni 10 file o all'ultimo
                if idx % 10 == 0 or idx == len(npz_files):
                    logger.info(f"Processed {idx}/{len(npz_files)} files.")
                    
            except Exception as e:
                logger.error(f"Error processing {npz_file}: {e}")
                continue  # Salta il file corrotto e continua

        logger.info(f"Successfully processed {processed_files}/{len(npz_files)} files.")


 
        # STEP 5: CONVERTI LE LISTE IN NUMPY ARRAY

      
        # Le liste sono state usate per appending efficiente
        # Ora convertiamo in numpy array per operazioni veloci
        for set_name in eeg_data.keys():
            for label in eeg_data[set_name].keys():
                if eeg_data[set_name][label]:
                    eeg_data[set_name][label] = np.array(eeg_data[set_name][label])
                else:
                    eeg_data[set_name][label] = np.array([])  # Array vuoto
                    logger.warning(f"No samples for {set_name} label {label}")



        # STEP 6: LOG DELLE STATISTICHE DEL DATASET

        total_train = sum(len(eeg_data['train'][label]) for label in range(5))
        total_test = sum(len(eeg_data['test'][label]) for label in range(5))
        
        logger.info("=" * 60)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total samples - Train: {total_train}, Test: {total_test}")
        logger.info("-" * 40)
        logger.info("Per-class distribution:")
        
        for label in range(5):
            train_count = len(eeg_data['train'][label])
            test_count = len(eeg_data['test'][label])
            label_names = ['W', 'N1', 'N2', 'N3', 'REM']
            logger.info(f"  {label_names[label]}: Train={train_count:6d} ({train_count/total_train*100:5.1f}%), "
                       f"Test={test_count:6d} ({test_count/total_test*100:5.1f}%)")
        
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

    return eeg_data



"""
# NOTE SULLO SPLIT DEI DATI
#
# Perché 85% train / 15% test?
#   - Il dataset Sleep-EDF ha 15 soggetti totali
#   - 85% = 13 soggetti per training
#   - 15% = 2 soggetti per test
#
#   - Questo file si occupa solo di train/test split finale, non vlidation set
#
# Struttura dei dati Sleep-EDF-2018:
#   - Ogni soggetto ha circa 20-30 epoch per stadio del sonno
#   - Ogni epoch = 30 secondi di segnale EEG a 100Hz = 3000 campioni
#   - Canale Fpz-Cz (frontale-centrale) è standard per sleep staging



caricato il file .nrz come li preprocessa, cioe è generico o specifivco per eeg

struttura npz 
- 'x': array di forma (N_epochs, 1, 3000) contenente gli epoch EEG
- 'y': array di forma (N_epochs,) contenente le etichette delle classi

prelaborazione tra import e pretask

leggere aticolo per capire dove mettono i pesi



gradi di liberta del codice

fissare chen ho capito blocchi codice 

una presentazione su come funziana il codice 

e vedere .npz  se è generico o specifico per eeg

"""
