import torch
import os

# Modifica questi path con i tuoi file
encoder_path = "checkpoints/encoder/encoder_1113.pth"
classifier_path = "checkpoints/classifier/classifier_1113.pth"

def analizza_modello(path, nome):
    if not os.path.exists(path):
        print(f"{nome}: File non trovato -> {path}")
        return
    
    state_dict = torch.load(path, map_location='cpu')
    num_params = sum(p.numel() for p in state_dict.values())
    file_size_kb = os.path.getsize(path) / 1024
    
    print(f"{nome}:")
    print(f"  Parametri: {num_params:,}")
    print(f"  Peso su disco: {file_size_kb:.2f} KB ({file_size_kb/1024:.2f} MB)")
    print()

# Esegui
analizza_modello(encoder_path, "ENCODER")
analizza_modello(classifier_path, "CLASSIFIER")

# Totale
if os.path.exists(encoder_path) and os.path.exists(classifier_path):
    e = sum(p.numel() for p in torch.load(encoder_path, map_location='cpu').values())
    c = sum(p.numel() for p in torch.load(classifier_path, map_location='cpu').values())
    print(f"TOTALE: {e + c:,} parametri")