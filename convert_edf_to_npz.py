import os
import numpy as np
import mne
from glob import glob

input_dir = "dset/Sleep-EDF-2018/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
output_dir = "dset/Sleep-EDF-2018/npz/Fpz-Cz"

os.makedirs(output_dir, exist_ok=True)

psg_files = glob(os.path.join(input_dir, "*-PSG.edf"))
print(f"Trovati {len(psg_files)} file PSG")

for psg_file in psg_files:
    basename = os.path.basename(psg_file)
    subject_code = basename[:6]
    
    hyp_files = glob(os.path.join(input_dir, f"{subject_code}*-Hypnogram.edf"))
    
    if not hyp_files:
        print(f"Hypnogram non trovato per {psg_file}")
        continue
    
    hyp_file = hyp_files[0]
    print(f"Elaborazione: {basename}")
    
    try:
        raw = mne.io.read_raw_edf(psg_file, preload=True)
        annot = mne.read_annotations(hyp_file)
        raw.set_annotations(annot)
        
        channel = None
        for ch in raw.ch_names:
            if 'Fpz-Cz' in ch or 'FpzCz' in ch:
                channel = ch
                break
        if channel is None:
            channel = raw.ch_names[0]
        
        eeg_data = raw.get_data(picks=[channel])[0]
        sfreq = raw.info['sfreq']
        epoch_length = int(30 * sfreq)
        
        n_epochs = len(eeg_data) // epoch_length
        epochs = []
        labels = []
        
        for i in range(n_epochs):
            start = i * epoch_length
            end = start + epoch_length
            epoch = eeg_data[start:end]
            t_start = start / sfreq
            
            stage = -1
            for ann in annot:
                if ann['onset'] <= t_start < ann['onset'] + ann['duration']:
                    desc = ann['description']
                    if desc == 'Sleep stage W':
                        stage = 0
                    elif desc == 'Sleep stage N1':
                        stage = 1
                    elif desc == 'Sleep stage N2':
                        stage = 2
                    elif desc == 'Sleep stage N3':
                        stage = 3
                    elif desc == 'Sleep stage R':
                        stage = 4
                    break
            
            if stage != -1:
                epochs.append(epoch)
                labels.append(stage)
        
        if epochs:
            epochs = np.array(epochs)
            labels = np.array(labels)
            output_file = os.path.join(output_dir, f"{subject_code}.npz")
            np.savez(output_file, x=epochs, y=labels)
            print(f"Salvato: {subject_code}.npz - {len(epochs)} epoche")
        else:
            print(f"Nessuna epoca valida")
            
    except Exception as e:
        print(f"Errore con {basename}: {e}")

print(f"Conversione completata! File salvati in {output_dir}")
