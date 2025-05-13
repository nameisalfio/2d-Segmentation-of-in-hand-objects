import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def get_clip_directories(dataset_path, max_clips=None):
    """
    Ottiene le directory dei clip dal percorso del dataset.
    
    Args:
        dataset_path: Percorso alla directory contenente i clip (es. train_aria)
        max_clips: Numero massimo di clip da processare (None per processare tutti)
    
    Returns:
        Lista dei percorsi alle directory dei clip
    """
    # Trova tutte le directory nel percorso del dataset
    clip_dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) 
                if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('clip-')]
    
    clip_dirs.sort()  # Ordina le directory
    
    if max_clips is not None and max_clips != 'all':
        clip_dirs = clip_dirs[:int(max_clips)]
    
    print(f"Trovati {len(clip_dirs)} clip in {dataset_path}")
    
    return clip_dirs

def decode_rle(rle_data, height, width):
    """
    Decodifica una maschera in formato RLE.
    
    Il formato RLE in HOT3D consiste in coppie [index, count] dove:
    - index è l'indice diretto del pixel nell'immagine appiattita
    - count è il numero di pixel da attivare consecutivamente
    
    Args:
        rle_data: Lista di coppie RLE [index, count]
        height: Altezza dell'immagine
        width: Larghezza dell'immagine
    
    Returns:
        Maschera binaria (numpy array)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(0, len(rle_data), 2):
        pixel_idx = rle_data[i]
        count = rle_data[i+1]
        
        for j in range(count):
            curr_y = (pixel_idx + j) // width
            curr_x = (pixel_idx + j) % width
            
            if 0 <= curr_y < height and 0 <= curr_x < width:
                mask[curr_y, curr_x] = 1
    
    return mask

def calculate_iou(box1, box2):
    """
    Calcola Intersection over Union (IoU) tra due bounding box.
    
    Args:
        box1: Box [x1, y1, x2, y2] o [x, y, w, h]
        box2: Box [x1, y1, x2, y2] o [x, y, w, h]
    
    Returns:
        Valore IoU
    """
    # Converti da [x, y, w, h] a [x1, y1, x2, y2] se necessario
    if len(box1) == 4:
        if box1[2] < box1[0] or box1[3] < box1[1]:  # È in formato [x, y, w, h]
            box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    if len(box2) == 4:
        if box2[2] < box2[0] or box2[3] < box2[1]:  # È in formato [x, y, w, h]
            box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    
    # Calcola intersezione
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def is_object_in_hand(obj_data, hands_data, camera_id):
    """
    Determina se un oggetto è in mano basandosi sui criteri del dataset HOT3D.
    
    Args:
        obj_data: Dati dell'oggetto dal file objects.json
        hands_data: Dati delle mani dal file hands.json
        camera_id: ID della telecamera
    
    Returns:
        True se l'oggetto è in mano, False altrimenti
    """
    # Verifica che ci siano mani nel frame
    if "left" not in hands_data and "right" not in hands_data:
        return False
    
    # 1. Verifica sovrapposizione delle maschere (se disponibili)
    if "masks_modal" in obj_data and camera_id in obj_data["masks_modal"]:
        obj_mask_data = obj_data["masks_modal"][camera_id]
        obj_mask = decode_rle(obj_mask_data["rle"], obj_mask_data["height"], obj_mask_data["width"])
        
        # Crea maschera combinata delle mani
        hand_mask = None
        
        for hand_type in ["left", "right"]:
            if hand_type in hands_data and "masks" in hands_data[hand_type] and camera_id in hands_data[hand_type]["masks"]:
                hand_mask_data = hands_data[hand_type]["masks"][camera_id]
                curr_hand_mask = decode_rle(hand_mask_data["rle"], hand_mask_data["height"], hand_mask_data["width"])
                
                if hand_mask is None:
                    hand_mask = curr_hand_mask
                else:
                    # Unione delle maschere di entrambe le mani
                    hand_mask = np.logical_or(hand_mask, curr_hand_mask).astype(np.uint8)
        
        # Se abbiamo maschere delle mani, verifica sovrapposizione
        if hand_mask is not None:
            # Calcola intersezione delle maschere
            intersection = np.logical_and(obj_mask, hand_mask).sum()
            if intersection > 0:
                # Se c'è almeno un pixel di sovrapposizione, considera l'oggetto in mano
                return True
    
    # 2. Verifica distanza 3D (se disponibile)
    if "T_world_from_object" in obj_data:
        obj_position = np.array(obj_data["T_world_from_object"]["translation_xyz"])
        
        min_distance = float('inf')
        for hand_type in ["left", "right"]:
            if hand_type in hands_data:
                hand_position = None
                
                # Prova mano_pose
                if "mano_pose" in hands_data[hand_type] and "wrist_xform" in hands_data[hand_type]["mano_pose"]:
                    hand_position = np.array(hands_data[hand_type]["mano_pose"]["wrist_xform"][3:6])
                
                # Prova umetrack_pose
                elif "umetrack_pose" in hands_data[hand_type] and "wrist_xform" in hands_data[hand_type]["umetrack_pose"]:
                    # Estrai la posizione dalla matrice 4x4
                    wrist_xform = np.array(hands_data[hand_type]["umetrack_pose"]["wrist_xform"])
                    hand_position = wrist_xform[:3, 3] if wrist_xform.shape == (4, 4) else None
                
                if hand_position is not None:
                    distance = np.linalg.norm(obj_position - hand_position)
                    min_distance = min(min_distance, distance)
        
        # Considerare in mano se la distanza è inferiore a 1 cm (0.01 m)
        if min_distance < 0.01:
            return True
    
    # 3. Metodo fallback basato su IoU tra bounding box
    if "boxes_amodal" in obj_data and camera_id in obj_data["boxes_amodal"]:
        obj_box = obj_data["boxes_amodal"][camera_id]
        
        # Ottieni i box delle mani
        hand_boxes = []
        for hand_type in ["left", "right"]:
            if hand_type in hands_data and "boxes_amodal" in hands_data[hand_type] and camera_id in hands_data[hand_type]["boxes_amodal"]:
                hand_boxes.append(hands_data[hand_type]["boxes_amodal"][camera_id])
        
        # Verifica IoU con ogni mano
        for hand_box in hand_boxes:
            # Standardizza i box
            if len(obj_box) == 4 and obj_box[2] < 100 and obj_box[3] < 100:  # [x, y, w, h]
                obj_x1, obj_y1, obj_w, obj_h = obj_box
                obj_box = [obj_x1, obj_y1, obj_x1 + obj_w, obj_y1 + obj_h]
            
            if len(hand_box) == 4 and hand_box[2] < 100 and hand_box[3] < 100:  # [x, y, w, h]
                hand_x1, hand_y1, hand_w, hand_h = hand_box
                hand_box = [hand_x1, hand_y1, hand_x1 + hand_w, hand_y1 + hand_h]
            
            # Calcola IoU
            iou = calculate_iou(obj_box, hand_box)
            if iou > IOU_THRESHOLD:
                return True
    
    return False

def save_debug_image(image, mask, output_path=None):
    """
    Salva un'immagine con sovrapposizione della maschera per debug.
    
    Args:
        image: Immagine originale (RGB)
        mask: Maschera binaria
        output_path: Percorso dove salvare l'immagine
    """
    if output_path is None:
        return
    
    # Crea copia dell'immagine
    vis_image = image.copy()
    
    # Assicurati che l'immagine sia RGB
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
    
    # Colore e opacità della maschera
    mask_color = np.array(BLUE_MASK_COLOR)
    alpha = MASK_ALPHA
    
    # Applica la maschera
    mask_bool = mask > 0
    if np.any(mask_bool):
        # Crea un'immagine con solo la maschera colorata
        mask_overlay = np.zeros_like(vis_image)
        mask_overlay[mask_bool] = mask_color
        
        # Sovrapponi con trasparenza
        vis_image[mask_bool] = ((1-alpha) * vis_image[mask_bool] + 
                              alpha * mask_overlay[mask_bool]).astype(np.uint8)
    
    # Salva l'immagine
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

