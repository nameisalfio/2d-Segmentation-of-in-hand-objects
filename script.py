import numpy as np
import matplotlib.pyplot as plt
import cv2
from config import *

# Carica 5 campioni casuali dal dataset di training
train_data = np.load(TRAIN_CACHE_PATH, allow_pickle=True).tolist()
sample_indices = np.random.choice(len(train_data), 5, replace=False)

for idx in sample_indices:
    sample = train_data[idx]
    print(f"Campione {idx}:")
    print(f"  Clip: {sample['clip_name']}, Frame: {sample['frame_id']}, Camera: {sample['camera_id']}")
    print(f"  Oggetti in mano: {len(sample['objects_in_hand'])}")
    
    # Controlla se il file debug esiste
    debug_dir = os.path.join(DEBUG_OUTPUT_DIR, sample['clip_name'])
    debug_file = os.path.join(debug_dir, f"{sample['frame_id']:06d}_{sample['camera_id']}_debug.jpg")
    
    if os.path.exists(debug_file):
        print(f"  File debug: {debug_file}")