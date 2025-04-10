import os
import json
import numpy as np
import cv2
import glob
import argparse
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from data.utils import get_clip_directories, decode_rle, is_object_in_hand, save_debug_image, clean_directory

def process_frame(frame_prefix, clip_path, camera_id, debug_dir=None):
    """
    Processa un singolo frame per una specifica telecamera, estraendo solo le immagini RGB
    e le maschere degli oggetti in mano.
    
    Args:
        frame_prefix: Prefisso del nome del file (es. "000000")
        clip_path: Percorso al clip
        camera_id: ID della telecamera
        debug_dir: Directory per il debug output (se None, non salva immagini)
    
    Returns:
        Dizionario con i dati processati o None in caso di errore
    """
    # Percorsi ai file
    img_file = os.path.join(clip_path, f"{frame_prefix}.image_{camera_id}.jpg")
    objects_file = os.path.join(clip_path, f"{frame_prefix}.objects.json")
    hands_file = os.path.join(clip_path, f"{frame_prefix}.hands.json")
    
    # Verifica che tutti i file necessari esistano
    if not all(os.path.exists(f) for f in [img_file, objects_file, hands_file]):
        return None
    
    # Carica l'immagine
    image = cv2.imread(img_file)
    if image is None:
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Carica i dati degli oggetti
    with open(objects_file, 'r') as f:
        objects_data = json.load(f)
    
    # Carica i dati delle mani
    with open(hands_file, 'r') as f:
        hands_data = json.load(f)
    
    # Crea una maschera combinata di tutti gli oggetti in mano
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    has_objects_in_hand = False
    
    for obj_id, obj_data_list in objects_data.items():
        if not obj_data_list or not isinstance(obj_data_list, list):
            continue
        
        obj_data = obj_data_list[0]
        obj_data["object_id"] = obj_id  # Aggiungi l'ID dell'oggetto ai dati
        
        # Verifica se l'oggetto ha maschera per questa telecamera
        if "masks_modal" in obj_data and camera_id in obj_data["masks_modal"]:
            # Determina se l'oggetto è in mano
            in_hand = is_object_in_hand(obj_data, hands_data, camera_id)
            
            # Includi solo oggetti in mano
            if in_hand:
                # Ottieni la maschera
                mask_data = obj_data["masks_modal"][camera_id]
                mask = decode_rle(mask_data["rle"], mask_data["height"], mask_data["width"])
                
                # Aggiungi alla maschera combinata (OR logico)
                combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
                has_objects_in_hand = True
    
    # Salva immagine di debug se richiesto
    if debug_dir and has_objects_in_hand:
        debug_path = os.path.join(debug_dir, f"{frame_prefix}_{camera_id}_mask.jpg")
        save_debug_image(image, combined_mask, debug_path)
    
    # Crea il record del frame solo se ci sono oggetti in mano
    if has_objects_in_hand:
        frame_data = {
            "frame_id": frame_prefix,
            "camera_id": camera_id,
            "image_path": img_file,
            "mask": combined_mask,        # Maschera combinata di tutti gli oggetti in mano
            "image_shape": (height, width)
        }
        return frame_data
    
    return None  # Ritorna None se non ci sono oggetti in mano

def process_clip(clip_path, camera_ids=USE_CAMERAS, debug=False):
    """
    Processa un intero clip e genera dati per tutte le telecamere e frame.
    
    Args:
        clip_path: Percorso al clip
        camera_ids: Lista di ID telecamera da processare
        debug: Se True, salva immagini di debug
    
    Returns:
        Lista di frame processati
    """
    clip_name = os.path.basename(clip_path)
    print(f"Processando clip: {clip_name}")
    
    processed_frames = []
    
    # Crea directory di debug
    debug_dir = None
    if debug:
        debug_dir = os.path.join(DEBUG_OUTPUT_DIR, "annotated", clip_name)
        os.makedirs(debug_dir, exist_ok=True)
    
    # Trova tutti i prefissi dei frame
    frame_prefixes = sorted(set([
        os.path.basename(f).split('.')[0] 
        for f in glob.glob(os.path.join(clip_path, "*.image_*.jpg"))
    ]))
    
    # Processa ogni frame per ogni telecamera
    for camera_id in camera_ids:
        for frame_prefix in tqdm(frame_prefixes, desc=f"Processando camera {camera_id}"):
            frame_data = process_frame(frame_prefix, clip_path, camera_id, debug_dir)
            if frame_data:
                # Aggiungi informazioni sul clip
                frame_data["clip_name"] = clip_name
                processed_frames.append(frame_data)
    
    return processed_frames

def process_dataset(dataset_type, max_clips=None, camera_ids=USE_CAMERAS, debug=False):
    """
    Processa l'intero dataset (train o test) e genera un unico file .npy.
    
    Args:
        dataset_type: 'train' o 'test'
        max_clips: Numero massimo di clip da processare ('all' per tutti)
        camera_ids: Lista di ID telecamera da processare
        debug: Se True, salva immagini di debug
    
    Returns:
        Percorso al file dei dati processati
    """
    # Determina i percorsi in base al tipo di dataset
    if dataset_type.lower() == 'train':
        dataset_path = os.path.join(HOT3D_DATASET_PATH, "train_aria")
        output_file = TRAIN_CACHE_PATH
    else:
        dataset_path = os.path.join(HOT3D_DATASET_PATH, "test_aria")
        output_file = TEST_CACHE_PATH
    
    # Assicurati che la directory di output esista (senza sottodirectory per clip)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Pulisci directory esistenti
    clean_directory(os.path.dirname(output_file))
    
    # Ottieni le directory dei clip
    clip_dirs = get_clip_directories(dataset_path, max_clips=max_clips)
    
    all_processed_frames = []
    
    # Processa ogni clip
    for clip_dir in tqdm(clip_dirs, desc=f"Processando clip {dataset_type}"):
        processed_frames = process_clip(clip_dir, camera_ids, debug)
        all_processed_frames.extend(processed_frames)
    
    # Salva tutti i dati in un unico file
    if all_processed_frames:
        np.save(output_file, all_processed_frames)
        print(f"Salvati in totale {len(all_processed_frames)} frame in {output_file}")
        
        # Se richiesto, crea anche il file di validation (80/20 split)
        if dataset_type.lower() == 'train':
            # Crea l'indice casuale
            np.random.seed(RANDOM_SEED)
            indices = np.random.permutation(len(all_processed_frames))
            val_size = int(len(all_processed_frames) * 0.2)
            val_indices = indices[:val_size]
            
            # Seleziona i frame per la validation
            val_frames = [all_processed_frames[i] for i in val_indices]
            
            # Salva il file di validation
            np.save(VAL_CACHE_PATH, val_frames)
            print(f"Salvati {len(val_frames)} frame in {VAL_CACHE_PATH}")
        
        return output_file
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Preprocessing del dataset HOT3D')
    parser.add_argument('dataset_type', choices=['train', 'test'], help='Tipo di dataset da processare')
    parser.add_argument('max_clips', help='Numero massimo di clip da processare (o "all" per tutti)')
    parser.add_argument('--debug', action='store_true', help='Salva immagini di debug')
    parser.add_argument('--cameras', nargs='+', default=USE_CAMERAS, help='ID telecamere da processare')
    
    args = parser.parse_args()
    
    process_dataset(args.dataset_type, args.max_clips, args.cameras, args.debug)

if __name__ == "__main__":
    main()