"""
ANALISI STRUTTURA DATASET SLEEP-EDF
"""

import numpy as np
import glob
import os

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

DATA_PATH = "/Users/riccardosasu/Desktop/simplesleepnet/dset/Sleep-EDF-2018/npz/Fpz-Cz"

# ============================================================================
# ANALISI COMPLETA DEL DATASET
# ============================================================================

print("=" * 80)
print("STRUTTURA DATASET SLEEP-EDF")
print("=" * 80)

# Trova tutti i file .npz
npz_files = sorted(glob.glob(os.path.join(DATA_PATH, "*.npz")))
print(f"\n📁 FILE TROVATI: {len(npz_files)}")

# Carica il primo file per analizzare la struttura
sample_file = npz_files[0]
sample_data = np.load(sample_file)

print("\n📁 STRUTTURA DI UN FILE NPZ:")
print(f"   File: {os.path.basename(sample_file)}")
print(f"   Chiavi nel file: {list(sample_data.keys())}")

print("\n" + "=" * 80)
print("DETTAGLI PER CHIAVE (primo file)")
print("=" * 80)

for key in sample_data.keys():
    arr = sample_data[key]
    print(f"\n🔍 {key}:")
    print(f"   Shape: {arr.shape}")
    print(f"   Tipo: {arr.dtype}")
    print(f"   Dimensione in memoria: {arr.nbytes / 1024:.2f} KB")
    
    if len(arr.shape) == 1:
        print(f"   Valori unici: {np.unique(arr)}")
        print(f"   Prime 10 etichette: {arr[:10]}")
    else:
        print(f"   Min: {arr.min():.4f}")
        print(f"   Max: {arr.max():.4f}")
        print(f"   Media: {arr.mean():.4f}")
        print(f"   Std: {arr.std():.4f}")
        print(f"   Prime 2 righe (prime 10 colonne):\n{arr[:2, :10]}")

print("\n" + "=" * 80)
print("STATISTICHE COMPLETE SU TUTTI I FILE")
print("=" * 80)

# Analisi aggregata su tutti i file
total_epochs = 0
total_files = len(npz_files)
subject_epochs = {}
subject_files = {}

for npz_file in npz_files:
    basename = os.path.basename(npz_file)
    subject = int(basename[3:5])
    
    with np.load(npz_file) as data:
        x = data['x']
        y = data['y']
        n_epochs = x.shape[0]
        
        total_epochs += n_epochs
        
        if subject not in subject_epochs:
            subject_epochs[subject] = 0
            subject_files[subject] = 0
        
        subject_epochs[subject] += n_epochs
        subject_files[subject] += 1

print(f"\n📊 RIEPILOGO GENERALE:")
print(f"   File totali: {total_files}")
print(f"   Epoche totali: {total_epochs}")
print(f"   Soggetti unici: {len(subject_epochs)}")
print(f"   Epoche medie per soggetto: {total_epochs / len(subject_epochs):.0f}")

print("\n📊 SOGGETTI:")
for subj in sorted(subject_epochs.keys()):
    print(f"   Soggetto {subj:2}: {subject_files[subj]} file, {subject_epochs[subj]:5} epoche")

print("\n" + "=" * 80)
print("INTERPRETAZIONE FORMATO DATI")
print("=" * 80)

# Carica un file per l'interpretazione
with np.load(sample_file) as data:
    x = data['x']
    y = data['y']

print(f"\n📊 FORMATO DATI:")

# Gestisce shape 2D o 3D
if len(x.shape) == 2:
    print(f"   Shape x: (epoche, campioni) = {x.shape}")
    print(f"   → {x.shape[0]} epoche (segmenti da 30 secondi)")
    print(f"   → {x.shape[1]} campioni per epoca")
    n_canali = 1
    n_campioni = x.shape[1]
else:
    print(f"   Shape x: (epoche, canali, campioni) = {x.shape}")
    print(f"   → {x.shape[0]} epoche (segmenti da 30 secondi)")
    print(f"   → {x.shape[1]} canale/i EEG")
    print(f"   → {x.shape[2]} campioni per epoca")
    n_canali = x.shape[1]
    n_campioni = x.shape[2]

print(f"\n🎯 FORMATO ETICHETTE:")
print(f"   Shape y: (epoche,) = {y.shape}")
print(f"   Classi presenti: {np.unique(y)}")
print(f"   Significato classi:")
print(f"      0 = W (Wake) - Veglia")
print(f"      1 = N1 (NREM1) - Sonno leggero")
print(f"      2 = N2 (NREM2) - Sonno intermedio")
print(f"      3 = N3 (NREM3) - Sonno profondo (onde lente)")
print(f"      4 = REM (REM) - Sonno paradossale")

print(f"\n⏱️  FREQUENZA CAMPIONAMENTO:")
print(f"   Campioni per epoca: {n_campioni}")
print(f"   Durata epoca: 30 secondi")
print(f"   Frequenza: {n_campioni / 30:.0f} Hz")

print("\n📈 DISTRIBUZIONE ETICHETTE (nel primo file):")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    label_names = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    print(f"   {label_names[label]}: {count} epoche ({count/len(y)*100:.1f}%)")

print("\nANALISI COMPLETATA")