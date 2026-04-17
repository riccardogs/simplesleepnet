import glob
import numpy as np
import os
import matplotlib.pyplot as plt

NPZ_PATH = 'dset/Sleep-EDF-2018/npz/Fpz-Cz'
npz_files = sorted(glob.glob(os.path.join(NPZ_PATH, '*.npz')))

subject_stats = {}

for f in npz_files:
    basename = os.path.basename(f)
    subject = int(basename[3:5])
    
    with np.load(f) as data:
        y = data['y']
        unique_labels = set(y)
        
        if subject not in subject_stats:
            subject_stats[subject] = {'files': 0, 'epochs': 0, 'labels': set()}
        
        subject_stats[subject]['files'] += 1
        subject_stats[subject]['epochs'] += len(y)
        subject_stats[subject]['labels'].update(unique_labels)

print('SOGGETTO | FILE | EPOCHE | LABELS PRESENTI')
print('-' * 50)
for subj in sorted(subject_stats.keys()):
    stats = subject_stats[subj]
    labels_str = ','.join([str(l) for l in sorted(stats['labels'])])
    print(f'  {subj:4}   |  {stats["files"]:2}   |  {stats["epochs"]:5}   |  {labels_str}')

# ============================================================================
# VISUALIZZA SOLO I CAMBI DI FASE (transizioni)
# ============================================================================

def visualizza_transizioni(subject_idx):
    """Carica il soggetto e mostra solo le epoche dove cambia lo stadio."""
    
    subject_files = [f for f in npz_files if int(os.path.basename(f)[3:5]) == subject_idx]
    
    if not subject_files:
        print(f"Soggetto {subject_idx} non trovato")
        return
    
    all_x = []
    all_y = []
    
    for f in subject_files:
        with np.load(f) as data:
            all_x.append(data['x'])
            all_y.append(data['y'])
    
    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    LABELS = {0: 'W (Wake)', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    COLORS = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red', 4: 'purple'}
    
    n_epochs = len(y)
    
    # Trova i punti dove cambia la label (transizioni)
    transitions = []
    for i in range(1, n_epochs):
        if y[i] != y[i-1]:
            transitions.append(i)
    
    print(f"\n📊 SOGGETTO {subject_idx}")
    print(f"   Totale epoche: {n_epochs}")
    print(f"   Transizioni trovate: {len(transitions)}")
    print("\n📍 EPOCHE DI TRANSIZIONE (cambio stadio):")
    print("   Indice | Da → A")
    print("   " + "-" * 30)
    for idx in transitions[:50]:  # stampa solo prime 50 per non affollare
        print(f"   {idx:5} | {LABELS[y[idx-1]]} → {LABELS[y[idx]]}")
    if len(transitions) > 50:
        print(f"   ... e altre {len(transitions)-50} transizioni")
    
    # Mostra la sequenza degli stadi
    print("\n📈 SEQUENZA DEGLI STADI:")
    current_label = y[0]
    start_idx = 0
    sequences = []
    for i in range(1, n_epochs):
        if y[i] != current_label:
            sequences.append((start_idx, i-1, current_label))
            start_idx = i
            current_label = y[i]
    sequences.append((start_idx, n_epochs-1, current_label))
    
    for start, end, label in sequences[:30]:  # mostra solo prime 30 sequenze
        duration = (end - start + 1) * 30
        print(f"   Epoche {start:4} → {end:4} : {LABELS[label]:10} ({duration} secondi = {duration/60:.1f} minuti)")
    if len(sequences) > 30:
        print(f"   ... e altre {len(sequences)-30} sequenze")
    
    # ========================================================================
    # PLOT DEI CAMBI DI FASE
    # ========================================================================
    
    n_transitions = len(transitions)
    if n_transitions == 0:
        print("\n⚠️ Nessuna transizione trovata (un solo stadio per tutto il soggetto)")
        return
    
    max_plot = min(n_transitions, 12)
    
    fig, axes = plt.subplots(max_plot, 1, figsize=(15, 3 * max_plot))
    if max_plot == 1:
        axes = [axes]
    
    time_axis = np.linspace(0, 30, 3000)
    
    for plot_idx, trans_idx in enumerate(transitions[:max_plot]):
        if len(x.shape) == 3:
            signal_before = x[trans_idx - 1, 0, :]
            signal_after = x[trans_idx, 0, :]
        else:
            signal_before = x[trans_idx - 1, :]
            signal_after = x[trans_idx, :]
        
        label_before = y[trans_idx - 1]
        label_after = y[trans_idx]
        
        ax = axes[plot_idx]
        ax.plot(time_axis, signal_before, color=COLORS[label_before], linewidth=0.7, alpha=0.7)
        ax.plot(time_axis, signal_after, color=COLORS[label_after], linewidth=0.7, alpha=0.7)
        ax.set_title(f'Transizione epoca {trans_idx}: {LABELS[label_before]} → {LABELS[label_after]}', fontsize=10)
        ax.set_xlim(0, 30)
        ax.set_ylabel('µV')
        ax.grid(True, alpha=0.3)
        ax.legend([f'Prima ({LABELS[label_before]})', f'Dopo ({LABELS[label_after]})'], fontsize=8)
    
    axes[-1].set_xlabel('Tempo (secondi)')
    plt.suptitle(f'Soggetto {subject_idx} - Transizioni tra stadi del sonno', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # PLOT DI UN EPOCA PER OGNI STADIO (IN VERTICALE)
    # ========================================================================
    
    # Trova quali stadi sono effettivamente presenti
    stadi_presenti = sorted(set(y))
    
    print(f"\n📊 STADI PRESENTI NEL SOGGETTO {subject_idx}: {[LABELS[s] for s in stadi_presenti]}")
    
    # Crea una figura con tanti subplot quanti sono gli stadi presenti (verticale)
    n_stadi = len(stadi_presenti)
    fig2, axes2 = plt.subplots(n_stadi, 1, figsize=(15, 4 * n_stadi))
    
    if n_stadi == 1:
        axes2 = [axes2]
    
    time_axis = np.linspace(0, 30, 3000)
    
    for i, label in enumerate(stadi_presenti):
        indices = np.where(y == label)[0]
        if len(indices) > 0:
            epoch_idx = indices[0]
            if len(x.shape) == 3:
                signal = x[epoch_idx, 0, :]
            else:
                signal = x[epoch_idx, :]
            
            axes2[i].plot(time_axis, signal, color=COLORS[label], linewidth=0.7)
            axes2[i].set_title(f'{LABELS[label]} - prima epoca trovata: epoca {epoch_idx}', fontsize=11)
            axes2[i].set_xlim(0, 30)
            axes2[i].set_ylabel('µV')
            axes2[i].grid(True, alpha=0.3)
    
    axes2[-1].set_xlabel('Tempo (secondi)')
    plt.suptitle(f'Soggetto {subject_idx} - Prima epoca per ogni stadio presente ({n_stadi} stadi)', fontsize=14)
    plt.tight_layout()
    plt.show()

# ============================================================================
# SCEGLI IL SOGGETTO DA ANALIZZARE
# ============================================================================

visualizza_transizioni(1)