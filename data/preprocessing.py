import os
import json
import numpy as np
import cv2
import glob
from tqdm import tqdm
import sys
from pycocotools import mask as mask_utils # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from data.utils import save_debug_image

def process_frame(image_info, annotations, categories, dataset_path, debug_dir=None):
    """
    Processa un singolo frame dal dataset VISOR, estraendo oggetti in mano.
    """
    image_id = image_info["id"]
    image_path = os.path.join(dataset_path, "images", image_info["file_name"])
    
    # Verifica che l'immagine esista
    if not os.path.exists(image_path):
        print(f"Immagine non trovata: {image_path}")
        return None
    
    # Carica l'immagine
    image = cv2.imread(image_path)
    if image is None:
        print(f"Impossibile leggere l'immagine: {image_path}")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Filtra le annotazioni per questa immagine
    image_annotations = [ann for ann in annotations if ann["image_id"] == image_id]
    
    # Ottieni le categorie
    hand_category_id = None
    object_category_id = None
    for cat in categories:
        if "hand" in cat["name"].lower():
            hand_category_id = cat["id"]
        else:
            object_category_id = cat["id"]
    
    # Filtra solo le annotazioni delle mani in contatto con oggetti
    hands_in_contact = [ann for ann in image_annotations if 
                      ann["category_id"] == hand_category_id and 
                      ann.get("isincontact", 0) == 1]
    
    # Filtra solo le annotazioni degli oggetti
    objects = [ann for ann in image_annotations if ann["category_id"] == object_category_id]
    
    if not hands_in_contact or not objects:
        return None  # Nessuna mano in contatto con oggetti o nessun oggetto in questo frame
    
    # Associamo le mani agli oggetti basandoci sulla sovrapposizione spaziale
    objects_in_hand = []
    for hand in hands_in_contact:
        hand_box = hand["bbox"]
        hand_x1, hand_y1, hand_w, hand_h = hand_box
        hand_x2, hand_y2 = hand_x1 + hand_w, hand_y1 + hand_h
        
        for obj in objects:
            obj_box = obj["bbox"]
            obj_x1, obj_y1, obj_w, obj_h = obj_box
            obj_x2, obj_y2 = obj_x1 + obj_w, obj_y1 + obj_h
            
            # Verifica sovrapposizione
            if (max(hand_x1, obj_x1) < min(hand_x2, obj_x2) and 
                max(hand_y1, obj_y1) < min(hand_y2, obj_y2)):
                # C'è sovrapposizione, aggiungiamo questo oggetto alla lista
                if obj not in objects_in_hand:
                    objects_in_hand.append(obj)
    
    if not objects_in_hand:
        return None  # Nessun oggetto in mano in questo frame
    
    # Crea una maschera combinata di tutti gli oggetti in mano
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Liste per raccogliere maschere e box individuali
    individual_masks = []
    individual_boxes = []
    individual_labels = []
    individual_object_ids = []
    
    for obj in objects_in_hand:
        # Estrai la maschera di segmentazione
        segmentation = obj.get("segmentation", [])
        
        if not segmentation:
            continue
            
        # La segmentazione può essere in formato RLE o poligono
        mask = None
        if isinstance(segmentation, dict):  # RLE
            mask = mask_utils.decode(segmentation)
        elif isinstance(segmentation, list) and len(segmentation) > 0:  # Poligono
            mask = np.zeros((height, width), dtype=np.uint8)
            if isinstance(segmentation[0], list):  # Lista di poligoni
                for poly_points in segmentation:
                    poly = np.array(poly_points, np.int32).reshape((-1, 2))
                    cv2.fillPoly(mask, [poly], 1)
            elif isinstance(segmentation[0], (int, float)):  # Un solo poligono appiattito
                poly = np.array(segmentation, np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [poly], 1)
        
        if mask is None:
            continue
            
        # Estrai il bounding box [x, y, width, height]
        box = obj["bbox"]
        
        # Converti in formato [x1, y1, x2, y2]
        x1, y1, w, h = box
        box_xyxy = [float(x1), float(y1), float(x1 + w), float(y1 + h)]
        
        # Ottieni la categoria
        category_id = obj["category_id"]
        
        # Aggiungi alla maschera combinata
        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
        
        # Salva le informazioni individuali
        individual_masks.append(mask)
        individual_boxes.append(box_xyxy)
        individual_labels.append(category_id)
        individual_object_ids.append(obj["id"])
    
    # Salva immagine di debug se richiesto
    if debug_dir and len(individual_masks) > 0:
        # Crea un'immagine che mostri sia i box che le maschere
        debug_vis_path = os.path.join(debug_dir, f"image_{image_id}_vis.jpg")
        vis_image = image.copy()
        
        # Applica maschere colorate semi-trasparenti
        mask_overlay = np.zeros_like(vis_image)
        mask_color = (0, 0, 255)
        
        for i, mask in enumerate(individual_masks):
            mask_bool = mask > 0
            mask_overlay[mask_bool] = mask_color
            
            # Aggiungi anche il box corrispondente
            x1, y1, x2, y2 = [int(b) for b in individual_boxes[i]]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), mask_color, 2)
            
            # Aggiungi etichetta con ID categoria
            label = individual_labels[i]
            cv2.putText(vis_image, f"ID: {label}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, mask_color, 2)
        
        # Sovrapponi maschere con trasparenza
        alpha = 0.4  # Livello di trasparenza
        mask_bool = combined_mask > 0
        vis_image[mask_bool] = ((1-alpha) * vis_image[mask_bool] + 
                              alpha * mask_overlay[mask_bool]).astype(np.uint8)
        
        cv2.imwrite(debug_vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    # Crea il record del frame
    frame_data = {
        "image_id": image_id,
        "file_name": image_info["file_name"],
        "image_path": image_path,
        "mask": combined_mask,
        "individual_masks": individual_masks,
        "boxes": individual_boxes,
        "labels": individual_labels,
        "object_ids": individual_object_ids,
        "image_shape": (height, width)
    }
    
    return frame_data

def process_visor_dataset(json_file, dataset_path=HOT3D_DATASET_PATH, dataset_type='train', debug=False):
    """
    Processa il dataset VISOR da un file JSON e genera un file .npy.
    
    Args:
        json_file: Percorso al file JSON con le annotazioni
        dataset_path: Percorso base del dataset
        dataset_type: Tipo di output ('train', 'val', 'test')
        debug: Se True, salva immagini di debug
    
    Returns:
        Percorso al file dei dati processati
    """
    print(f"Elaborazione del file JSON: {json_file}")
    
    # Carica il file JSON con le annotazioni
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Estrai le informazioni rilevanti
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    
    print(f"Trovate {len(images)} immagini, {len(annotations)} annotazioni e {len(categories)} categorie")
    print(f"Categorie disponibili: {[(cat['id'], cat['name']) for cat in categories]}")
    
    # Determina il file di output
    if dataset_type == 'train':
        output_file = TRAIN_CACHE_PATH
    elif dataset_type == 'val':
        output_file = VAL_CACHE_PATH
    else:
        output_file = TEST_CACHE_PATH
    
    # Assicurati che la directory di output esista
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Crea directory di debug
    debug_dir = None
    if debug:
        debug_dir = os.path.join(DEBUG_OUTPUT_DIR, "visor_annotations")
        os.makedirs(debug_dir, exist_ok=True)
    
    processed_frames = []
    
    # Processa ogni immagine
    for image_info in tqdm(images, desc="Processando immagini"):
        frame_data = process_frame(image_info, annotations, categories, dataset_path, debug_dir)
        if frame_data:
            processed_frames.append(frame_data)
    
    # Salva tutti i dati in un file
    if processed_frames:
        np.save(output_file, processed_frames)
        print(f"Salvati in totale {len(processed_frames)} frame in {output_file}")
        return output_file
    
    return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocessing del dataset VISOR')
    parser.add_argument('--dataset', choices=['train_25', 'val', 'test'], default='train',
                      help='Tipo di dataset (train, val o test)')
    parser.add_argument('--debug', action='store_true', help='Salva immagini di debug')
    args = parser.parse_args()
    
    # Processa il dataset
    json_file = os.path.join(HOT3D_DATASET_PATH, os.path.join("annotations", f"{args.dataset}.json"))
    process_visor_dataset(json_file, HOT3D_DATASET_PATH, args.dataset, args.debug)

if __name__ == "__main__":
    main()