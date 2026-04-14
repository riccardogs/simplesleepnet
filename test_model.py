import torch
import numpy as np
import glob
import random
from models.simple_sleep_net import SimpleSleepNet
from models.sleep_stage_classifier import SleepStageClassifier

# Carica modello
encoder = SimpleSleepNet(latent_dim=64, dropout=0.2)
encoder.load_state_dict(torch.load('checkpoints/encoder/encoder_10119.pth', map_location='cpu'))
encoder.eval()

classifier = SleepStageClassifier(input_dim=64, num_classes=5, dropout_probs=0.5)
classifier.load_state_dict(torch.load('checkpoints/classifier/classifier_10119.pth', map_location='cpu'))
classifier.eval()

# Carica file
npz_files = sorted(glob.glob('./dset/Sleep-EDF-2018/npz/Fpz-Cz/*.npz'))
file_path = npz_files[0]
data = np.load(file_path)

class_names = ['W', 'N1', 'N2', 'N3', 'REM']




classe_da_testare = 2 

# Cerca un file che contiene la classe scelta
npz_files = sorted(glob.glob('./dset/Sleep-EDF-2018/npz/Fpz-Cz/*.npz'))

file_con_n1 = None
for file_path in npz_files:
    data = np.load(file_path)
    if np.sum(data['y'] == classe_da_testare ) > 0:
        file_con_n1 = file_path
        break

if file_con_n1 is None:
    print(f"Nessun file {class_names[classe_da_testare]} trovato!")
    exit()

data = np.load(file_con_n1)
class_names = ['W', 'N1', 'N2', 'N3', 'REM']

# Trova tutti i segmenti N1
n1_indices = np.where(data['y'] == classe_da_testare)[0]

if len(n1_indices) > 0:
    indice = np.random.choice(n1_indices)
    segnale = data['x'][indice]
    label_reale = data['y'][indice]
    
    print(f"File: {file_con_n1.split('/')[-1]}")
    print(f"Segmento {indice}: etichetta reale = {class_names[label_reale]}")
    
    input_tensor = torch.tensor(segnale, dtype=torch.float).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        embedding = encoder(input_tensor)
        output = classifier(embedding)
        prob = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
    
    print(f"\n PREDIZIONE:")
    print(f" Reale:    {class_names[label_reale]}")
    print(f" Predetto: {class_names[prediction]}")
    print(f"\n Probabilità:")
    for i, name in enumerate(class_names):
        print(f"    {name}: {prob[0][i]:.2f}")
else:
    print(f"Nessun {class_names[classe_da_testare]} trovato")