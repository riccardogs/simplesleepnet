"""
1. Preparazione dei dati (caricamento e split train/test)
2. Pretraining contrastivo dell'encoder (self-supervised)
3. Valutazione dello spazio latente (t-SNE, UMAP)
4. Training del classificatore supervisionato (fine-tuning)
5. Test finale e salvataggio dei risultati
6. Tracciamento dei tempi di esecuzione


Il pipeline segue l'approccio SimCLR:
- Fase 1: Apprendimento contrastivo senza etichette
- Fase 2: Classificazione supervisionata con encoder congelato
"""

import sys
import os
import argparse
import json
import logging
import time
from datetime import timedelta

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import ContrastiveEEGDataset, SupervisedEEGDataset
from utils import load_eeg_data, validate_config, set_seed, setup_logging, setup_tensorboard, get_tensorboard_logger, close_tensorboard
from models import SimpleSleepNet, SleepStageClassifier
from training import train_contrastive_model, train_classifier
from evaluation import LatentSpaceEvaluator, get_predictions, ResultsSaver
from augmentations import load_augmentations_from_config


def suppress_warnings():
    """Sopprime warning non critici per mantenere l'output pulito."""
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

suppress_warnings()

NUM_CLASSES = 5  # W, N1, N2, N3, REM


def parse_args():
    """
    Parsing dei comandi da riga di terminale.
    
    Parsing = analisi e interpretazione di dati strutturati.

        Nel contesto del codice:
        - Prende la stringa del comando da terminale (es. "--config config.json")
        - La suddivide in parti riconoscibili
        - La trasforma in una struttura dati (oggetto) che il programma può usare

    Returns:
        argparse.Namespace: Contiene --config (percorso file configurazione)
                           e --list-configs (lista file disponibili)
    """
    parser = argparse.ArgumentParser(description='Sleep Stage Classification')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default/config.json',
        help='Path to the config file. Example: configs/experiment1/config1.json'
    )
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List all available configuration files and exit.'
    )
    return parser.parse_args()



def load_config(config_path):
    """
    Carica il file JSON di configurazione.
    
    Args:
        config_path (str): Percorso del file .json
        
    Returns:
        dict: Configurazione parsata
        
    Exits:
        Se file non trovato o JSON non valido
    """
    if not os.path.isfile(config_path):
        logging.error(f"Configuration file '{config_path}' not found.")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from the config file: {e}")
        sys.exit(1)




def list_available_configs(configs_dir='configs'):
    """
    Scansiona la directory configs/ e stampa tutti i file .json disponibili.
    Utile per vedere quali configurazioni si possono lanciare.
    """
    print("Available configuration files:")
    for root, dirs, files in os.walk(configs_dir):
        for file in files:
            if file.endswith('.json'):
                config_path = os.path.join(root, file)
                print(config_path)




def setup_environment(config):
    """
    Configura l'ambiente di esecuzione:
    - Directory per i log
    - Seed per riproducibilità
    - Device (CUDA, MPS, o CPU)
    - TensorBoard per il monitoraggio
    
    Args:
        config (dict): Configurazione
        
    Returns:
        tuple: (logger, device, tensorboard_logger)
    """
    # Crea directory per i log se non esiste
    os.makedirs('logs', exist_ok=True)
    
    # Configura il logging: sia su file che su console
    setup_logging(log_level=logging.INFO, log_file=f'logs/experiment_{config["experiment_num"]}.log')
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Starting the EEG Project")
    logger.info("=" * 80)
    
    # Imposta seed per riproducibilità
    set_seed(config["seed"])
    logger.info(f"Random seed set to {config['seed']}")
    
    # Rilevazione automatica del dispositivo
    # Priorità: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS backend for acceleration")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (no GPU detected)")
    
    # Crea directory per TensorBoard
    os.makedirs(f'runs/experiment_{config["experiment_num"]}', exist_ok=True)
    setup_tensorboard(log_dir=f'runs/experiment_{config["experiment_num"]}')
    tensorboard_logger = get_tensorboard_logger()
    logger.info(f"TensorBoard logging initialized at: runs/experiment_{config['experiment_num']}")
    
    return logger, device, tensorboard_logger




def prepare_datasets(config, logger):
    """
    Prepara i dataset e i dataloader per il training supervisionato.
    
    IMPORTANTE: Lo split train/test viene fatto UNA VOLTA da load_eeg_data().
    train_loader contiene i soggetti di training (85% = 12 soggetti)
    test_loader contiene i soggetti di test (15% = 3 soggetti)
    
    Args:
        config (dict): Configurazione
        logger: Logger
        
    Returns:
        tuple: (eeg_data, train_loader, test_loader)
    """

    BATCH_SIZE = config["pretraining_params"]["batch_size"]
    NUM_WORKERS = config["num_workers"]
    
    """
        BATCH_SIZE = numero di campioni elaborati contemporaneamente dal modello.

        Esempio con BATCH_SIZE=256:
        - Il modello prende 256 segnali EEG
        - Calcola gli embeddings per tutti e 256 in parallelo
        - Aggiorna i pesi UNA VOLTA dopo aver visto 256 campioni

        NUM_WORKERS = numero di processi che caricano i dati in parallelo.

        Esempio con NUM_WORKERS=8:
        - 8 processi separati leggono i file NPZ contemporaneamente
        - Preparano i batch mentre il modello sta elaborando
        - Riduce i tempi di attesa per i dati

                PERCHÉ SONO IMPORTANTI:

                BATCH_SIZE grande:
                - ✅ Più stabile (gradiente più accurato)
                - ✅ Più veloce (parallelismo)
                - ❌ Richiede più memoria GPU/RAM

                BATCH_SIZE piccolo:
                - ✅ Meno memoria
                - ❌ Gradiente più rumoroso
                - ❌ Più lento

                NUM_WORKERS alto:
                - ✅ Caricamento dati più veloce
                - ❌ Più consumo CPU
                - ❌ Overhead di comunicazione

        VALORI TIPICI NEL TUO CODICE:
        - BATCH_SIZE: 256 o 512
        - NUM_WORKERS: 8 (nel tuo config)
    """ 
    
    # Carica i dati e fa lo split per soggetti
    eeg_data = load_eeg_data(
        dataset_path=config['dataset']['dset_path'],
        num_files_to_process=config['dataset']['max_files']
    )
    logger.info("Loaded train and test sets of EEG data")

    # Crea dataset supervisionati (segnale, label)
    train_dataset = SupervisedEEGDataset(eeg_data['train'])
    test_dataset = SupervisedEEGDataset(eeg_data['test'])

    # DataLoader: shuffle=True per training, shuffle=False per test
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    logger.info("Supervised datasets and dataloaders created.")
    
    return eeg_data, train_loader, test_loader

"""
quindi qui:

Hai già i dati divisi in:
- eeg_data['train'] = 12 soggetti (85%)
- eeg_data['test'] = 3 soggetti (15%)

ORA devi prepararli per essere usati dal modello.

1. SupervisedEEGDataset(eeg_data['train'])
   → Trasforma i dati grezzi in un formato che PyTorch capisce
   → Ogni elemento = (segnale_EEG, label)
   → Esempio: ( [0.5, 0.3, ...], 2 )  segnale + etichetta N2

2. DataLoader(...)
   → Prende il dataset e lo suddivide in "pacchetti" (batch)
   → Ogni volta che chiedi un batch, il DataLoader ti dà BATCH_SIZE esempi
   → Gestisce anche il caricamento in parallelo (num_workers)

shuffle=True per train:
   → Mescola i campioni ad ogni epoca
   → Evita che il modello impari l'ordine dei dati

shuffle=False per test:
   → Non mescola (l'ordine non importa)
   → Riproducibile

RISULTATO FINALE:

train_loader = ti dà batch di (256 segnali, 256 labels) dal training set
test_loader = ti dà batch di (256 segnali, 256 labels) dal test set

QUINDI:

Dataset → DataLoader → Batch pronti per il modello
"""








def pretrain_contrastive_model(config, eeg_data, device, logger, tensorboard_logger):
    """
    FASE 1: ADDESTRAMENTO CONTRASTIVO DELL'ENCODER
    
    L'encoder impara rappresentazioni utili SENZA usare le etichette.
    Vengono generate due views augmentate dello stesso segnale.
    La loss NT-Xent avvicina le views dello stesso segnale e allontana quelle diverse.
    
    CRITICO: validation = eeg_data['test'] (3 soggetti) ❌
    Il validation set dovrebbe essere un sottoinsieme del training set.
    
    Args:
        config (dict): Configurazione
        eeg_data (dict): Dati già suddivisi in train/test
        device: torch.device
        logger: Logger
        tensorboard_logger: Logger per TensorBoard
        
    Returns:
        SimpleSleepNet: Encoder addestrato
    """
    BATCH_SIZE = config["pretraining_params"]["batch_size"]
    LATENT_DIM = config["pretraining_params"]["latent_dim"]
    DROP_PROB = config["pretraining_params"]["dropout_rate"]
    NUM_WORKERS = config["num_workers"]
    TEMP = config["pretraining_params"]["temperature"]
    
    # Carica le augmentations dal config (es. TimeWarping, RandomNoise, etc.)
    augmentations = load_augmentations_from_config(config=config)

    # Dataset di training contrastivo (genera coppie di views augmentate)
    train_contrastive_dataset = ContrastiveEEGDataset(eeg_data['train'], augmentations=augmentations)
    train_contrastive_loader = DataLoader(
        train_contrastive_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    logger.info(f"Contrastive train dataset created with {len(train_contrastive_dataset)} samples")

    # Dataset di validation contrastivo
    # PROBLEMA: usa eeg_data['test'] invece di un sottoinsieme del training
    val_contrastive_dataset = ContrastiveEEGDataset(eeg_data['test'], augmentations=augmentations)
    val_contrastive_loader = DataLoader(
        val_contrastive_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    logger.info(f"Contrastive test dataset created with {len(val_contrastive_dataset)} samples")

    # Inizializza encoder
    encoder = SimpleSleepNet(latent_dim=LATENT_DIM, dropout=DROP_PROB).to(device)
    
    # Log dell'architettura su TensorBoard
    sample_input = torch.zeros(1, 1, 3000).to(device)
    tensorboard_logger.add_graph(encoder, sample_input)
    
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params} trainable parameters ({total_params * 4 / 1024:.2f} KB)")

    # Ottimizzatore Adam
    contrastive_optimizer = optim.Adam(encoder.parameters(), lr=config["pretraining_params"]["learning_rate"])
    
    # Crea directory per i checkpoint
    best_encoder_pth = f"{config['pretraining_params']['best_model_pth']}{config['experiment_num']}.pth"
    os.makedirs(os.path.dirname(best_encoder_pth), exist_ok=True)

    # Avvia training contrastivo
    train_contrastive_model(
        model=encoder,
        dataloader=train_contrastive_loader,
        optimizer=contrastive_optimizer,
        device=device,
        num_epochs=config["pretraining_params"]["max_epochs"],
        temperature=TEMP,
        val_dataloader=val_contrastive_loader,  # ← validation = test set
        check_interval=config["pretraining_params"]["check_interval"],
        min_improvement=config["pretraining_params"]["min_improvement"],
        best_model_path=best_encoder_pth
    )
    logger.info("Contrastive training complete")
    
    # Carica i pesi migliori
    try:
        encoder.load_state_dict(torch.load(best_encoder_pth, map_location=device))
        logger.info("Loaded best encoder from %s", best_encoder_pth)
    except Exception as e:
        logger.error("Error loading best encoder: %s", str(e))
        raise

    return encoder





"""

VALIDATION SET - A COSA SERVE

Il validation set è un insieme di dati SEPARATO che viene usato DURANTE il training per prendere decisioni, 
ma NON per aggiornare i pesi del modello.

UTILIZZI PRINCIPALI:

1. EARLY STOPPING:
   - Monitora la loss sul validation set ad ogni epoca
   - Quando la loss smette di migliorare, ferma il training
   - Previene overfitting (il modello impara a memoria i dati di training)

2. SALVATAGGIO DEL MIGLIOR MODELLO:
   - Salva il modello quando la performance sul validation set migliora
   - Non salva l'ultimo modello, ma quello che generalizza meglio

3. REGOLAZIONE IPERPARAMETRI:
   - Scegliere learning rate, dropout, etc. in base alla performance su validation
   - Testare diverse configurazioni

4. PREVENIRE DATA LEAKAGE:
   - Il test set rimane intoccato fino alla fine
   - Solo validation set viene usato per decisioni durante training

"""






def evaluate_latent_space(config, encoder, eeg_data, device, logger):
    """
    FASE 2: VALUTAZIONE DELLO SPAZIO LATENTE
    
    Visualizza gli embeddings usando t-SNE o UMAP per verificare
    se i segnali dello stesso tipo si raggruppano insieme.
    
    Args:
        config (dict): Configurazione
        encoder: Modello addestrato
        eeg_data (dict): Dati EEG
        device: torch.device
        logger: Logger
    """
    # Salta se disabilitato nella configurazione
    if not config["latent_space_params"].get("tsne_enabled", False) and \
       not config["latent_space_params"].get("umap_enabled", False):
        logger.info("Latent space visualization disabled, skipping...")
        return
        
    BATCH_SIZE = config["pretraining_params"]["batch_size"]
    NUM_WORKERS = config["num_workers"]
    
    # Usa il test set per la visualizzazione
    visualization_dataset = ContrastiveEEGDataset(
        eeg_signals=eeg_data['test'],
        augmentations=[],
        return_labels=True
    )
    visualization_loader = DataLoader(
        visualization_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    evaluator = LatentSpaceEvaluator(
        model=encoder,
        dataloader=visualization_loader,
        device=device,
        umap_enabled=config["latent_space_params"]["umap_enabled"],
        pca_enabled=config["latent_space_params"]["pca_enabled"],
        tsne_enabled=config["latent_space_params"]["tsne_enabled"],
        visualize=config["latent_space_params"]["visualize"],
        compute_metrics=config["latent_space_params"]["compute_metrics"],
        n_clusters=config["latent_space_params"]["n_clusters"],
        output_image_dir=config["latent_space_params"]["output_image_dir"],
        output_metrics_dir=config["latent_space_params"]["output_metrics_dir"],
        experiment_num=config["experiment_num"],
        visualization_fraction=config["latent_space_params"]["visualization_fraction"]
    )
    evaluator.run()
    logger.info("Latent space evaluation complete")







def train_supervised_classifier(config, encoder, train_loader, test_loader, device, logger, tensorboard_logger):
    """
    FASE 3: ADDESTRAMENTO SUPERVISIONATO DEL CLASSIFICATORE
    
    L'encoder viene CONGELATO (requires_grad=False).
    Si addestra solo il classificatore MLP per predire i 5 stadi del sonno.
    
    CRITICO: validation = test_loader (3 soggetti) ❌
    Il validation set dovrebbe essere un sottoinsieme del training set.
    
    Args:
        config (dict): Configurazione
        encoder: Encoder pre-addestrato (frozen)
        train_loader: DataLoader per training
        test_loader: DataLoader per test (usato come validation)
        device: torch.device
        logger: Logger
        tensorboard_logger: Logger TensorBoard
        
    Returns:
        tuple: (classifier, best_classifier_pth)
    """
    LATENT_DIM = config["pretraining_params"]["latent_dim"]
    DROP_PROB = config["sup_training_params"]["dropout_rate"]
    
    # Inizializza classificatore MLP
    classifier = SleepStageClassifier(
        input_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        dropout_probs=DROP_PROB
    ).to(device)
    
    # Log dell'architettura su TensorBoard
    sample_input = torch.zeros(1, LATENT_DIM).to(device)
    tensorboard_logger.add_graph(classifier, sample_input)
    
    # Loss: CrossEntropy (include softmax internamente)
    criterion = nn.CrossEntropyLoss()
    supervised_optimizer = optim.Adam(classifier.parameters(), lr=config["sup_training_params"]["learning_rate"])
    
    total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    logger.info(f"Classifier created with {total_params} trainable parameters")

    # CONGELA L'ENCODER: i pesi non vengono aggiornati
    for param in encoder.parameters():
        param.requires_grad = False
    logger.info("Encoder frozen")

    # Crea directory per i checkpoint
    best_classifier_pth = config["sup_training_params"]["best_model_pth"] + str(config["experiment_num"]) + ".pth"
    os.makedirs(os.path.dirname(best_classifier_pth), exist_ok=True)
    
    # Avvia training del classificatore
    # PROBLEMA: val_loader = test_loader (usa il test set come validation)
    train_classifier(
        encoder=encoder,
        classifier=classifier,
        train_loader=train_loader,
        val_loader=test_loader,  # ← validation = test set ❌
        criterion=criterion,
        optimizer=supervised_optimizer,
        num_epochs=config["sup_training_params"]["max_epochs"],
        device=device,
        save_path=best_classifier_pth,
        check_interval=config["sup_training_params"]["check_interval"],
        min_improvement=config["sup_training_params"]["min_improvement"]
    )
    logger.info("Classifier training complete")
    
    return classifier, best_classifier_pth







def test_and_save_results(config, encoder, classifier, test_loader, device, logger):
    """
    FASE 4: TEST FINALE E SALVATAGGIO RISULTATI
    
    Valuta il modello sul test set e salva:
    - Matrice di confusione (numeri e percentuali)
    - Accuratezza e Macro F1
    - Metriche per classe (precision, recall, F1)
    - Plot della matrice di confusione
    
    Args:
        config (dict): Configurazione
        encoder: Encoder (frozen)
        classifier: Classificatore addestrato
        test_loader: DataLoader del test set
        device: torch.device
        logger: Logger
    """

    # Carica i pesi migliori del classificatore
    best_classifier_pth = f"{config['sup_training_params']['best_model_pth']}{config['experiment_num']}.pth"
    classifier.load_state_dict(torch.load(best_classifier_pth, map_location=device))
    logger.info(f"Loaded best classifier from {best_classifier_pth}")
    
    # Ottiene predizioni e label vere
    predictions, true_labels = get_predictions(encoder, classifier, test_loader, device=device)
    
    # Salva risultati nella sottocartella experiment_{num}
    results_saver = ResultsSaver(
        results_folder=config["results_folder"],
        experiment_num=config["experiment_num"]
    )
    results_saver.save_classification_results(
        predictions=predictions,
        true_labels=true_labels,
        num_classes=NUM_CLASSES
    )
    logger.info("Classification results saved")









def save_timing_file(config, timings):
    """
    Salva i tempi di esecuzione in un file TXT nella cartella dei risultati.
    
    Il file viene salvato in: results/experiment_{num}/execution_times_{num}.txt
    
    Args:
        config (dict): Configurazione
        timings (dict): Dizionario con i tempi di ogni fase
    """
    experiment_num = config.get("experiment_num", "default")
    
    # Crea la sottocartella experiment_{num}
    results_base = config.get("results_folder", "results")
    results_folder = os.path.join(results_base, f"experiment_{experiment_num}")
    os.makedirs(results_folder, exist_ok=True)
    
    filepath = os.path.join(results_folder, f"execution_times_{experiment_num}.txt")
    
    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"EXECUTION TIMES - Experiment {experiment_num}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("TIMING BREAKDOWN:\n")
        f.write("-" * 50 + "\n")
        
        for phase, duration in timings.items():
            phase_name = phase.replace('_', ' ').title()
            # Formatta in ore/minuti/secondi per durate lunghe
            if duration >= 3600:
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                f.write(f"{phase_name:30}: {duration:8.2f} seconds  ({hours}h {minutes}m {seconds}s)\n")
            elif duration >= 60:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                f.write(f"{phase_name:30}: {duration:8.2f} seconds  ({minutes}m {seconds}s)\n")
            else:
                f.write(f"{phase_name:30}: {duration:8.2f} seconds\n")
        
        f.write("\n" + "=" * 70 + "\n")
        total_secs = timings['total']
        hours = int(total_secs // 3600)
        minutes = int((total_secs % 3600) // 60)
        seconds = int(total_secs % 60)
        f.write(f"{'TOTAL EXECUTION TIME':30}: {total_secs:8.2f} seconds")
        f.write(f"  ({hours}h {minutes}m {seconds}s)\n")
        f.write("=" * 70 + "\n")
    
    print(f"✓ Timing saved to {filepath}")


def main():
    """
    MAIN: ESEGUE L'INTERO PIPELINE
    
    Flusso:
    1. Parsing argomenti
    2. Caricamento configurazione
    3. Setup ambiente
    4. Preparazione dati (split train/test)
    5. Pretraining contrastivo (encoder)
    6. Valutazione spazio latente
    7. Training classificatore (fine-tuning)
    8. Test e salvataggio risultati
    9. Salvataggio tempi
    """
    start_total = time.time()
    
    args = parse_args()

    if args.list_configs:
        list_available_configs()
        sys.exit(0)

    config = load_config(args.config)
    validate_config(config)

    logger, device, tensorboard_logger = setup_environment(config)
    
    # ========== FASE 1: PREPARAZIONE DATI ==========
    logger.info("-" * 60)
    logger.info("PHASE 1: Data Preparation")
    logger.info("-" * 60)
    start_prep = time.time()
    eeg_data, train_loader, test_loader = prepare_datasets(config, logger)
    prep_time = time.time() - start_prep
    logger.info(f"✓ Data preparation completed in {prep_time:.2f} seconds")
    
    # ========== FASE 2: PRETRAINING CONTRASTIVO ==========
    logger.info("-" * 60)
    logger.info("PHASE 2: Contrastive Pretraining")
    logger.info("-" * 60)
    start_ct = time.time()
    encoder = pretrain_contrastive_model(config, eeg_data, device, logger, tensorboard_logger)
    contrastive_time = time.time() - start_ct
    logger.info(f"✓ Contrastive training completed in {contrastive_time:.2f} seconds")
    
    # ========== FASE 3: VALUTAZIONE SPAZIO LATENTE ==========
    logger.info("-" * 60)
    logger.info("PHASE 3: Latent Space Evaluation")
    logger.info("-" * 60)
    start_le = time.time()
    evaluate_latent_space(config, encoder, eeg_data, device, logger)
    latent_eval_time = time.time() - start_le
    logger.info(f"✓ Latent space evaluation completed in {latent_eval_time:.2f} seconds")
    
    # ========== FASE 4: TRAINING CLASSIFICATORE ==========
    logger.info("-" * 60)
    logger.info("PHASE 4: Classifier Training")
    logger.info("-" * 60)
    start_cl = time.time()
    classifier, _ = train_supervised_classifier(config, encoder, train_loader, test_loader, device, logger, tensorboard_logger)
    classifier_time = time.time() - start_cl
    logger.info(f"✓ Classifier training completed in {classifier_time:.2f} seconds")
    
    # ========== FASE 5: TEST E SALVATAGGIO ==========
    logger.info("-" * 60)
    logger.info("PHASE 5: Testing and Results Saving")
    logger.info("-" * 60)
    start_test = time.time()
    test_and_save_results(config, encoder, classifier, test_loader, device, logger)
    test_time = time.time() - start_test
    logger.info(f"✓ Testing completed in {test_time:.2f} seconds")
    
    total_time = time.time() - start_total
    
    # Salva i tempi di esecuzione
    save_timing_file(
        config=config,
        timings={
            'data_preparation': prep_time,
            'contrastive_training': contrastive_time,
            'latent_space_evaluation': latent_eval_time,
            'classifier_training': classifier_time,
            'testing_and_saving': test_time,
            'total': total_time
        }
    )
    
    # Riepilogo finale
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info(f"Total execution time: {total_time:.2f} seconds ({int(total_time//3600)}h {int((total_time%3600)//60)}m {int(total_time%60)}s)")
    logger.info(f"Results saved to: {config['results_folder']}/experiment_{config['experiment_num']}/")
    logger.info("=" * 80)
    
    close_tensorboard()


if __name__ == "__main__":
    main()