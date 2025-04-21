import os
import sys
import argparse
import random
import glob
import cv2
import numpy as np
from tqdm import tqdm
import json

# Aggiungi la directory principale al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MaskRCNNModel
from config import MODEL_SAVE_PATH, NUM_CLASSES, RESULTS_DIR

def apply_mask(image, mask, color, alpha=0.5):
    """Applica una maschera colorata all'immagine"""
    colored_mask = np.zeros_like(image)
    colored_mask[:, :] = color
    
    mask_image = np.where(mask[:, :, None] > 0, 
                          cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0), 
                          image)
    
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_image, contours, -1, color, 2)
    
    return mask_image

def draw_predictions(image, predictions, class_names=None, score_threshold=0.5):
    """Disegna le predizioni sull'immagine"""
    masks = predictions['masks']
    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']
    
    if len(scores) == 0:
        return image.copy()
    
    output_image = image.copy()
    
    # Lista di colori per le diverse classi (BGR)
    colors = [
        (0, 255, 0),    # Verde
        (255, 0, 0),    # Blu
        (0, 0, 255),    # Rosso
        (0, 255, 255),  # Giallo
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Ciano
    ]
    
    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue
        
        color = colors[labels[i] % len(colors)]
        
        if masks is not None and len(masks) > 0:
            mask = masks[i]
            output_image = apply_mask(output_image, mask, color)

    return output_image

def process_hot3d_clip(model, clip_path, output_dir, class_names, num_frames=10, camera_id="214-1", threshold=0.5):
    """
    Processa un clip HOT3D selezionando un numero specificato di frame casuali
    
    Args:
        model: Modello Mask R-CNN
        clip_path: Percorso alla cartella del clip
        output_dir: Directory di output
        class_names: Nomi delle classi
        num_frames: Numero di frame casuali da selezionare
        camera_id: ID della telecamera
        threshold: Soglia di confidenza
    
    Returns:
        Dictionary con i risultati delle predizioni
    """
    clip_name = os.path.basename(clip_path)
    clip_output_dir = os.path.join(output_dir, clip_name)
    os.makedirs(clip_output_dir, exist_ok=True)
    
    # Trova tutti i file immagine per la camera specificata
    image_pattern = f"*.image_{camera_id}.jpg"
    all_images = glob.glob(os.path.join(clip_path, image_pattern))
    
    if not all_images:
        print(f"Nessuna immagine trovata per camera {camera_id} in {clip_path}")
        return {}
    
    # Seleziona casualmente un sottoinsieme di immagini
    if num_frames < len(all_images):
        random.seed(42)  # Per riproducibilità
        selected_images = random.sample(all_images, num_frames)
    else:
        selected_images = all_images
    
    results = {}
    
    for img_path in selected_images:
        frame_id = os.path.basename(img_path).split('.')[0]
        sample_id = f"{clip_name}_{frame_id}_{camera_id}"
        
        # Carica l'immagine
        image = cv2.imread(img_path)
        if image is None:
            print(f"Errore: Impossibile caricare l'immagine {img_path}")
            continue
        
        # Esegui la predizione
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictions = model.predict(image_rgb, score_threshold=threshold)
        
        # Genera l'immagine con le predizioni
        result_image = draw_predictions(image, predictions, class_names, threshold)
        
        # Salva l'immagine
        output_path = os.path.join(clip_output_dir, f"pred_{frame_id}_{camera_id}.png")
        cv2.imwrite(output_path, result_image)
        
        # Salva le maschere separate
        masks = []
        for i, mask in enumerate(predictions['masks']):
            if predictions['scores'][i] >= threshold:
                mask_path = os.path.join(clip_output_dir, f"{frame_id}_{camera_id}_mask_{i}.png")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                
                masks.append({
                    "mask_path": mask_path,
                    "score": float(predictions['scores'][i]),
                    "label": int(predictions['labels'][i]),
                    "box": predictions['boxes'][i].tolist()
                })
        
        # Salva i risultati
        results[sample_id] = {
            "image_path": img_path,
            "output_path": output_path,
            "masks": masks,
            "num_predictions": len(masks)
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Processa clip del dataset HOT3D')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory contenente le cartelle dei clip (es. ../hot3d/test_aria)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(RESULTS_DIR, "test_predictions"),
                        help='Directory di output per i risultati')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH,
                        help='Percorso del file del modello')
    parser.add_argument('--num_clips', type=int, default=100,
                        help='Numero massimo di clip da processare')
    parser.add_argument('--frames_per_clip', type=int, default=10,
                        help='Numero di frame casuali da selezionare per ogni clip')
    parser.add_argument('--camera_id', type=str, default="214-1",
                        help='ID della telecamera da utilizzare')
    parser.add_argument('--class_names', type=str, default="background,object",
                        help='Nomi delle classi separati da virgola')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Soglia di confidenza per le predizioni')
    
    args = parser.parse_args()
    
    # Verifica esistenza della directory di input
    if not os.path.exists(args.input_dir):
        print(f"ERRORE: Directory di input non trovata: {args.input_dir}")
        return
    
    # Verifica esistenza del file del modello
    if not os.path.exists(args.model):
        print(f"ERRORE: File del modello non trovato: {args.model}")
        return
    
    # Inizializza il modello
    print(f"Inizializzazione modello da {args.model}...")
    model = MaskRCNNModel(num_classes=NUM_CLASSES)
    
    # Carica il modello
    if not model.load(args.model):
        print(f"ERRORE: Impossibile caricare il modello da {args.model}")
        return
    
    # Prepara la directory di output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Trova tutte le cartelle dei clip
    clip_folders = [d for d in glob.glob(os.path.join(args.input_dir, "clip-*")) if os.path.isdir(d)]
    
    if not clip_folders:
        print(f"Nessuna cartella clip trovata in {args.input_dir}")
        return
    
    # Limita il numero di clip se necessario
    if args.num_clips < len(clip_folders):
        clip_folders = clip_folders[:args.num_clips]
    
    print(f"Processando {len(clip_folders)} clip, {args.frames_per_clip} frame casuali per clip")
    
    # Elabora ogni clip
    all_results = {}
    
    for clip_folder in tqdm(clip_folders, desc="Elaborazione clip"):
        clip_results = process_hot3d_clip(
            model, 
            clip_folder, 
            args.output_dir, 
            args.class_names.split(','), 
            args.frames_per_clip, 
            args.camera_id, 
            args.threshold
        )
        all_results.update(clip_results)
    
    # Salva tutti i risultati in un file JSON
    results_json_path = os.path.join(args.output_dir, "all_predictions.json")
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nProcessamento completato. Risultati salvati in {args.output_dir}")
    print(f"Riassunto salvato in {results_json_path}")
    print(f"Totale immagini elaborate: {len(all_results)}")
    
    # Conta le predizioni
    total_predictions = sum(result["num_predictions"] for result in all_results.values())
    print(f"Totale predizioni: {total_predictions}")
    
    # Calcola la media di predizioni per immagine
    if len(all_results) > 0:
        avg_predictions = total_predictions / len(all_results)
        print(f"Media predizioni per immagine: {avg_predictions:.2f}")

if __name__ == "__main__":
    main()