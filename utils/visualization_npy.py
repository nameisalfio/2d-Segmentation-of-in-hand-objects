#!/usr/bin/env python3
"""
Script per visualizzare le prime N immagini dal dataset di training e sovrapporre
la maschera dell'oggetto in mano.
"""

import os
import sys
import numpy as np
import cv2
import argparse
from tqdm import tqdm

# Aggiungi la directory principale al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def visualize_samples(dataset_path, output_dir, num_images=5):
    """
    Visualizza e salva le prime N immagini dal dataset sovrapposti con le maschere.
    
    Args:
        dataset_path: Percorso al file .npy del dataset
        output_dir: Directory dove salvare le visualizzazioni
        num_images: Numero di immagini da visualizzare (le prime N)
    """
    # Crea la directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Caricamento dataset da {dataset_path}...")
    
    try:
        # Carica l'intero dataset
        data = np.load(dataset_path, allow_pickle=True)
        print(f"Dataset caricato, dimensione totale: {len(data)} campioni.")
        
        # Limita il numero di immagini alla dimensione del dataset
        if num_images > len(data):
            num_images = len(data)
            print(f"Numero di immagini ridotto a {num_images} (dimensione del dataset).")
        
        # Prendi solo i primi N campioni
        data = data[:num_images]
        
    except Exception as e:
        print(f"Errore nel caricare il dataset: {str(e)}")
        return
    
    # Visualizza ogni campione
    for i, sample in enumerate(tqdm(data, desc="Visualizzando immagini")):
        # Carica l'immagine
        image_path = sample["image_path"]
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Impossibile leggere l'immagine: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Errore nel caricare l'immagine {image_path}: {str(e)}")
            continue
        
        # Ottieni la maschera
        mask = sample["mask"]
        
        # Assicurati che la maschera abbia le dimensioni corrette
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        # Crea una visualizzazione combinata: immagine originale, maschera, sovrapposizione
        
        # 1. Immagine originale
        orig_image = image.copy()
        
        # 2. Maschera a colori (per una migliore visualizzazione)
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [0, 0, 255]  # Colore blu per la maschera
        
        # 3. Immagine con maschera sovrapposta
        masked_image = image.copy()
        alpha = 0.5  # Opacità della maschera
        mask_bool = mask > 0
        masked_image[mask_bool] = (masked_image[mask_bool] * (1 - alpha) + 
                                  colored_mask[mask_bool] * alpha).astype(np.uint8)
        
        # Crea un'immagine composita con tutte e tre le visualizzazioni affiancate
        composite = np.concatenate([orig_image, colored_mask, masked_image], axis=1)
        
        # Aggiungi informazioni come testo
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # Bianco
        font_thickness = 1
        
        # Informazioni da aggiungere
        clip_name = sample.get("clip_name", "Unknown")
        frame_id = sample.get("frame_id", "Unknown")
        camera_id = sample.get("camera_id", "Unknown")
        
        info_text = f"Clip: {clip_name}, Frame: {frame_id}, Camera: {camera_id}"
        
        # Aggiungi il testo alla parte inferiore dell'immagine
        text_size = cv2.getTextSize(info_text, font, font_scale, font_thickness)[0]
        text_x = 10
        text_y = composite.shape[0] - 10  # 10 pixel dal basso
        
        # Aggiungi un rettangolo nero dietro il testo per leggibilità
        cv2.rectangle(composite, 
                     (text_x - 5, text_y - text_size[1] - 5), 
                     (text_x + text_size[0] + 5, text_y + 5), 
                     (0, 0, 0), 
                     -1)  # -1 per riempire il rettangolo
        
        cv2.putText(composite, 
                   info_text, 
                   (text_x, text_y), 
                   font, 
                   font_scale, 
                   font_color, 
                   font_thickness)
        
        # Aggiungi intestazioni per chiarire ciascuna immagine
        headers = ["Immagine originale", "Maschera", "Sovrapposizione"]
        segment_width = composite.shape[1] // 3
        
        for idx, header in enumerate(headers):
            text_size = cv2.getTextSize(header, font, font_scale, font_thickness)[0]
            text_x = idx * segment_width + (segment_width - text_size[0]) // 2
            text_y = 20  # 20 pixel dall'alto
            
            # Aggiungi rettangolo nero
            cv2.rectangle(composite, 
                         (text_x - 5, text_y - text_size[1] - 5), 
                         (text_x + text_size[0] + 5, text_y + 5), 
                         (0, 0, 0), 
                         -1)
            
            cv2.putText(composite, 
                       header, 
                       (text_x, text_y), 
                       font, 
                       font_scale, 
                       font_color, 
                       font_thickness)
        
        # Salva l'immagine composita
        output_file = os.path.join(output_dir, f"sample_{i+1:03d}.jpg")
        cv2.imwrite(output_file, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    
    print(f"Visualizzazioni salvate in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualizza le prime N immagini dal dataset con maschere')
    parser.add_argument('--dataset', choices=['train', 'val', 'test'], default='train',
                        help='Dataset da visualizzare (train, val o test)')
    parser.add_argument('--images', type=int, default=5,
                        help='Numero di immagini da visualizzare (le prime N)')
    parser.add_argument('--output_dir', default='visualizations',
                        help='Directory dove salvare le visualizzazioni')
    
    args = parser.parse_args()
    
    # Determina il percorso del dataset in base all'argomento
    if args.dataset == 'train':
        dataset_path = TRAIN_CACHE_PATH
    elif args.dataset == 'val':
        dataset_path = VAL_CACHE_PATH
    else:
        dataset_path = TEST_CACHE_PATH
    
    # Crea la directory di output completa
    output_dir = os.path.join(args.output_dir, args.dataset)
    
    # Visualizza i campioni
    visualize_samples(dataset_path, output_dir, args.images)

if __name__ == "__main__":
    main()