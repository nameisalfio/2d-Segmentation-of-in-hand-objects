import os

# Percorsi di base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("/storage/aspoto")
HOT3D_DATASET_PATH = os.path.join(HOME_DIR, "visor_egohos_synth") 

# Directory del progetto
PROJECT_DIR = os.path.join(HOME_DIR, "mask_rcnn_project")
MODELS_DIR = os.path.join(PROJECT_DIR, "saved_models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
DEBUG_OUTPUT_DIR = os.path.join(PROJECT_DIR, "debug_output")
DATASET_CACHE_DIR = os.path.join(PROJECT_DIR, "dataset_cache")
TENSORBOARD_DIR = os.path.join(PROJECT_DIR, "tensorboard_logs")

# Crea le directory se non esistono
for dir_path in [MODELS_DIR, RESULTS_DIR, DEBUG_OUTPUT_DIR, DATASET_CACHE_DIR, TENSORBOARD_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# File di cache del dataset (direttamente nella directory di cache, no sottodirectory)
TRAIN_CACHE_PATH = os.path.join(DATASET_CACHE_DIR, "train_dataset.npy")
VAL_CACHE_PATH = os.path.join(DATASET_CACHE_DIR, "val_dataset.npy")
TEST_CACHE_PATH = os.path.join(DATASET_CACHE_DIR, "test_dataset.npy")

TRAIN_CACHE_PATH = os.path.join(DATASET_CACHE_DIR, "train_dataset_reduced_1000.npy")
VAL_CACHE_PATH = os.path.join(DATASET_CACHE_DIR, "val_dataset_reduced_1000.npy")
TEST_CACHE_PATH = os.path.join(DATASET_CACHE_DIR, "test_dataset_reduced_100.npy")

# Parametri di addestramento
BATCH_SIZE = 2
NUM_EPOCHS = 500
LEARNING_RATE = 0.001
NUM_CLASSES = 2  # background + object_in_hand
RANDOM_SEED = 42

# Parametri del dataset HOT3D
USE_CAMERAS = ["214-1"]  # Utilizza tutte e tre le telecamere
TRAIN_CLIPS = [f"clip-{i:06d}" for i in range(1849, 2500)]  # Range di clip da utilizzare per training
VAL_CLIPS = [f"clip-{i:06d}" for i in range(2500, 2800)]    # Range di clip da utilizzare per validation
TEST_CLIPS = [f"clip-{i:06d}" for i in range(3365, 3832)]  # Range di clip da utilizzare per test

# Soglie per determinare se un oggetto è in mano
IOU_THRESHOLD = 0.1       # Soglia IoU tra mano e oggetto
DISTANCE_THRESHOLD = 0.01  # Soglia di distanza tra mano e oggetto (in metri)
VELOCITY_THRESHOLD = 0.01  # Soglia di velocità dell'oggetto (in m/s)

# Parametri RLE per le maschere
RLE_CONVERSION_FORMAT = "coco"  # Formato per conversione RLE

# Parametri di visualizzazione
BLUE_MASK_COLOR = [0, 0, 255]  # Blu per le maschere
MASK_ALPHA = 0.6  # Opacità delle maschere

# Percorso del modello salvato
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "mask_rcnn_final_visor.pth")