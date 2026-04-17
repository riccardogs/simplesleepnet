"""
VISUALIZZATORE DATASET EEG - Sleep-EDF
=======================================
Questo script permette di:
1. Elencare tutti i file .edf e .npz nella directory
2. Visualizzare le shape dei file .npz (N_epochs, canali, campioni)
3. Plottare un segnale EEG di un epoca scelta
4. Mostrare la distribuzione delle etichette
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

BASE_PATH = "/Users/riccardosasu/Desktop/simplesleepnet/dset/Sleep-EDF-2018"
NPZ_PATH = os.path.join(BASE_PATH, "npz/Fpz-Cz")

# Mappa etichette
LABELS = {0: "W (Wake)", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
COLORS = {0: "blue", 1: "orange", 2: "green", 3: "red", 4: "purple"}


# ============================================================================
# FUNZIONI PER I FILE NPZ
# ============================================================================

def list_npz_files(npz_path):
    """Elenca tutti i file .npz e mostra le shape."""
    npz_files = sorted(glob.glob(os.path.join(npz_path, "*.npz")))
    
    print("=" * 80)
    print(f"FILE .NPZ TROVATI: {len(npz_files)}")
    print("=" * 80)
    
    for npz_file in npz_files:
        basename = os.path.basename(npz_file)
        with np.load(npz_file) as data:
            # Stampa tutte le chiavi del file per capire la struttura
            print(f"\n📁 {basename}")
            print(f"   Chiavi nel file: {list(data.keys())}")
            
            for key in data.keys():
                arr = data[key]
                print(f"   {key}: shape = {arr.shape}, dtype = {arr.dtype}")
    
    return npz_files


def load_npz_subject(npz_path, subject_idx=None):
    """Carica tutti i file di un soggetto specifico."""
    npz_files = sorted(glob.glob(os.path.join(npz_path, "*.npz")))
    
    if subject_idx is not None:
        subject_files = []
        for f in npz_files:
            basename = os.path.basename(f)
            try:
                subj = int(basename[3:5])
                if subj == subject_idx:
                    subject_files.append(f)
            except:
                continue
        npz_files = subject_files
    
    if not npz_files:
        print(f"Nessun file trovato per soggetto {subject_idx}")
        return None, None
    
    all_x = []
    all_y = []
    
    for npz_file in npz_files:
        with np.load(npz_file) as data:
            # Prova diverse possibili chiavi
            if 'x' in data:
                all_x.append(data['x'])
            elif 'eeg' in data:
                all_x.append(data['eeg'])
            elif 'signal' in data:
                all_x.append(data['signal'])
            else:
                # Prende il primo array disponibile
                first_key = list(data.keys())[0]
                all_x.append(data[first_key])
            
            if 'y' in data:
                all_y.append(data['y'])
            elif 'label' in data:
                all_y.append(data['label'])
    
    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"\n📊 Soggetto {subject_idx}: {len(npz_files)} file, {x.shape[0]} epoche totali")
    print(f"   Shape x: {x.shape}")
    print(f"   Shape y: {y.shape}")
    
    return x, y


# ============================================================================
# FUNZIONI PER IL PLOT
# ============================================================================

def plot_epoch(x, y, epoch_idx, sfreq=100, title="EEG Signal"):
    """
    Plotta un singolo epoca.
    """
    # Gestisce diverse shape possibili
    if len(x.shape) == 3:
        signal = x[epoch_idx, 0, :]  # (N, canali, campioni)
    elif len(x.shape) == 2:
        signal = x[epoch_idx, :]  # (N, campioni)
    else:
        signal = x[epoch_idx]
    
    label = y[epoch_idx]
    
    # Durata in secondi (assumendo 100 Hz)
    n_samples = len(signal)
    duration = n_samples / sfreq
    time_axis = np.linspace(0, duration, n_samples)
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    ax.plot(time_axis, signal, 'b-', linewidth=0.7)
    ax.set_xlabel('Tempo (secondi)')
    ax.set_ylabel('Ampiezza (µV)')
    ax.set_title(f'{title} - Epoca {epoch_idx} - {LABELS.get(label, label)}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Epoca {epoch_idx}: {LABELS.get(label, label)}")
    print(f"   Min: {signal.min():.2f} µV")
    print(f"   Max: {signal.max():.2f} µV")
    print(f"   Media: {signal.mean():.2f} µV")
    print(f"   Std: {signal.std():.2f} µV")
    print(f"   Durata: {duration:.1f} secondi")
    print(f"   Campioni: {n_samples}")
    print(f"   Frequenza campionamento stimata: {n_samples/duration:.1f} Hz")


def plot_label_distribution(y):
    """Plotta la distribuzione delle etichette."""
    label_counts = Counter(y)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels_names = [LABELS[i] for i in sorted(label_counts.keys())]
    counts = [label_counts[i] for i in sorted(label_counts.keys())]
    colors_list = [COLORS[i] for i in sorted(label_counts.keys())]
    
    bars = ax.bar(labels_names, counts, color=colors_list, edgecolor='black')
    ax.set_xlabel('Stadi del sonno')
    ax.set_ylabel('Numero di epoche')
    ax.set_title('Distribuzione delle etichette nel dataset')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print("\n📊 Distribuzione etichette:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percent = count / len(y) * 100
        print(f"   {LABELS[label]}: {count} epoche ({percent:.1f}%)")


def plot_multiple_epochs(x, y, epoch_indices, sfreq=100):
    """Plotta più epoche in una figura."""
    n_plots = len(epoch_indices)
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots))
    
    if n_plots == 1:
        axes = [axes]
    
    for i, epoch_idx in enumerate(epoch_indices):
        if len(x.shape) == 3:
            signal = x[epoch_idx, 0, :]
        else:
            signal = x[epoch_idx, :]
        
        label = y[epoch_idx]
        n_samples = len(signal)
        duration = n_samples / sfreq
        time_axis = np.linspace(0, duration, n_samples)
        
        axes[i].plot(time_axis, signal, 'b-', linewidth=0.7)
        axes[i].set_ylabel('µV')
        axes[i].set_title(f'Epoca {epoch_idx} - {LABELS.get(label, label)}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, duration)
    
    axes[-1].set_xlabel('Tempo (secondi)')
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Esegue l'analisi completa del dataset."""
    
    print("=" * 80)
    print("VISUALIZZATORE DATASET SLEEP-EDF")
    print("=" * 80)
    
    # 1. Elenca file NPZ e mostra shape
    print("\n🔍 SCANSIONE FILE .NPZ...")
    npz_files = list_npz_files(NPZ_PATH)
    
    # 2. Carica un soggetto specifico (es. soggetto 0)
    print("\n" + "=" * 80)
    print("ANALISI SOGGETTO 0")
    print("=" * 80)
    
    x, y = load_npz_subject(NPZ_PATH, subject_idx=0)
    
    if x is not None:
        print(f"\n📊 Soggetto 0:")
        print(f"   Totale epoche: {x.shape[0]}")
        
        # 3. Distribuzione etichette
        plot_label_distribution(y)
        
        # 4. Plotta una epoca esempio
        print("\n📈 PLOT EPOCA ESEMPIO...")
        plot_epoch(x, y, epoch_idx=0, title="Soggetto 0 - Epoca 0")
        
        # 5. Plotta più epoche
        print("\n📈 PLOT MULTIPLE EPOCHE...")
        epochs_to_plot = []
        for label in range(5):
            indices = np.where(y == label)[0]
            if len(indices) > 0:
                epochs_to_plot.append(indices[0])
        
        if epochs_to_plot:
            plot_multiple_epochs(x, y, epochs_to_plot[:5])
    
    print("\n✅ Analisi completata!")


if __name__ == "__main__":
    main()