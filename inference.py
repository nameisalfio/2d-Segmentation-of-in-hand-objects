import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from config import MODEL_SAVE_PATH, VISUALIZATIONS_DIR, NUM_CLASSES
from models.mask_rcnn import MaskRCNNModel
from utils.visualization import visualize_prediction

def inference(image_path, threshold=0.5, output_path=None, model_path=None):
    """
    Esegue l'inferenza su una singola immagine con distinzione tra classi
    
    Args:
        image_path: Percorso dell'immagine di input
        threshold: Soglia di confidenza per le predizioni
        output_path: Percorso per salvare la visualizzazione
        model_path: Percorso del modello da utilizzare (opzionale, usa MODEL_SAVE_PATH come default)
    """
    # Controlla se l'immagine esiste
    if not os.path.exists(image_path):
        print(f"Errore: Immagine {image_path} non trovata")
        return
    
    # Usa il percorso del modello specificato o quello di default
    model_path = model_path or MODEL_SAVE_PATH
    
    # Inizializza il modello
    print("Inizializzazione del modello Mask R-CNN...")
    mask_rcnn = MaskRCNNModel(num_classes=NUM_CLASSES, pretrained=False)
    
    # Carica il modello addestrato
    if not os.path.exists(model_path):
        print(f"Errore: Modello addestrato non trovato in {model_path}")
        return
    
    mask_rcnn.load(model_path)
    print(f"Modello caricato da {model_path}")
    
    # Carica l'immagine
    print(f"Caricamento dell'immagine {image_path}...")
    image = Image.open(image_path).convert("RGB")
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    
    # Esegui la predizione
    print("Esecuzione dell'inferenza...")
    predictions = mask_rcnn.predict(image_tensor, score_threshold=threshold)
    
    # Visualizza i risultati per classi separate
    vis_images = mask_rcnn.visualize_predictions(image_tensor, predictions, threshold=threshold)
    
    # Salva la visualizzazione
    if output_path:
        # Determina la directory e il nome base del file
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # Crea figura per tutte le predizioni
        plt.figure(figsize=(14, 10))
        plt.imshow(vis_images['all'])
        plt.title("Tutte le predizioni")
        plt.axis('off')
        plt.tight_layout()
        all_path = os.path.join(output_dir, f"{base_name}_all.png")
        plt.savefig(all_path)
        plt.close()
        
        # Crea figura per oggetti non in mano (classe 1)
        plt.figure(figsize=(14, 10))
        plt.imshow(vis_images['not_in_hand'])
        plt.title("Oggetti non in mano (Classe 1)")
        plt.axis('off')
        plt.tight_layout()
        class1_path = os.path.join(output_dir, f"{base_name}_not_in_hand.png")
        plt.savefig(class1_path)
        plt.close()
        
        # Crea figura per oggetti in mano (classe 2)
        plt.figure(figsize=(14, 10))
        plt.imshow(vis_images['in_hand'])
        plt.title("Oggetti in mano (Classe 2)")
        plt.axis('off')
        plt.tight_layout()
        class2_path = os.path.join(output_dir, f"{base_name}_in_hand.png")
        plt.savefig(class2_path)
        plt.close()
        
        # Crea figura comparativa
        plt.figure(figsize=(18, 8))
        
        # Immagine originale
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Immagine Originale")
        plt.axis('off')
        
        # Oggetti non in mano
        plt.subplot(1, 3, 2)
        plt.imshow(vis_images['not_in_hand'])
        n_class1 = len(predictions['not_in_hand']['masks'])
        plt.title(f"Oggetti non in mano (n={n_class1})")
        plt.axis('off')
        
        # Oggetti in mano
        plt.subplot(1, 3, 3)
        plt.imshow(vis_images['in_hand'])
        n_class2 = len(predictions['in_hand']['masks'])
        plt.title(f"Oggetti in mano (n={n_class2})")
        plt.axis('off')
        
        # Salva la figura comparativa
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        plt.savefig(comparison_path)
        plt.close()
        
        print(f"Visualizzazioni salvate in:")
        print(f"  - {all_path}")
        print(f"  - {class1_path}")
        print(f"  - {class2_path}")
        print(f"  - {comparison_path}")
    
    # Stampa statistiche
    n_total = len(predictions['all']['masks'])
    n_not_in_hand = len(predictions['not_in_hand']['masks'])
    n_in_hand = len(predictions['in_hand']['masks'])
    
    print("\nStatistiche di inferenza:")
    print(f"Totale oggetti rilevati: {n_total}")
    print(f"Oggetti non in mano (Classe 1): {n_not_in_hand}")
    print(f"Oggetti in mano (Classe 2): {n_in_hand}")
    
    if n_not_in_hand > 0:
        avg_score_not_in_hand = predictions['not_in_hand']['scores'].mean().item()
        print(f"Score medio oggetti non in mano: {avg_score_not_in_hand:.4f}")
    
    if n_in_hand > 0:
        avg_score_in_hand = predictions['in_hand']['scores'].mean().item()
        print(f"Score medio oggetti in mano: {avg_score_in_hand:.4f}")
    
    return predictions, vis_images

if __name__ == "__main__":
    # Se eseguito direttamente, processa gli argomenti della riga di comando
    import argparse
    
    parser = argparse.ArgumentParser(description="Inferenza con Mask R-CNN per segmentazione di oggetti in mano")
    parser.add_argument("--image", required=True, help="Percorso dell'immagine di input")
    parser.add_argument("--threshold", type=float, default=0.5, help="Soglia di confidenza")
    parser.add_argument("--output", default=None, help="Percorso per salvare i risultati")
    parser.add_argument("--model", default=None, help="Percorso del modello da utilizzare")
    args = parser.parse_args()
    
    inference(
        image_path=args.image,
        threshold=args.threshold,
        output_path=args.output,
        model_path=args.model
    )