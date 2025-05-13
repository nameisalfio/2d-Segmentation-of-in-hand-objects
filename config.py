import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("/storage/aspoto") # User's home directory, specific to this setup
HOT3D_DATASET_PATH = os.path.join(HOME_DIR, "visor_egohos_synth") # Path to the primary dataset

# Project directories
PROJECT_DIR = os.path.join(HOME_DIR, "mask_rcnn_project")
MODELS_DIR = os.path.join(PROJECT_DIR, "saved_models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
DEBUG_OUTPUT_DIR = os.path.join(PROJECT_DIR, "debug_output")
TENSORBOARD_DIR = os.path.join(PROJECT_DIR, "tensorboard_logs_visor")

DATASET_CACHE_DIR = os.path.join(PROJECT_DIR, "dataset_cache_hot3d")
#DATASET_CACHE_DIR = os.path.join(PROJECT_DIR, "dataset_cache_visor") # Alternative cache dir

# Create directories if they don't exist
for dir_path in [MODELS_DIR, RESULTS_DIR, DEBUG_OUTPUT_DIR, DATASET_CACHE_DIR, TENSORBOARD_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Dataset cache files (directly in the cache directory, no subdirectory)
TRAIN_CACHE_PATH = os.path.join(DATASET_CACHE_DIR, "train_dataset.npy")
VAL_CACHE_PATH = os.path.join(DATASET_CACHE_DIR, "val_dataset.npy")
TEST_CACHE_PATH = os.path.join(DATASET_CACHE_DIR, "test_dataset.npy")

# Training parameters
BATCH_SIZE = 2
NUM_EPOCHS = 500
LEARNING_RATE = 0.001
NUM_CLASSES = 2  # background + object_in_hand
RANDOM_SEED = 42

# HOT3D dataset parameters (or similar dataset structure)
USE_CAMERAS = ["214-1"]  # Specific cameras to use
TRAIN_CLIPS = [f"clip-{i:06d}" for i in range(1849, 2500)]
VAL_CLIPS = [f"clip-{i:06d}" for i in range(2500, 2800)]
TEST_CLIPS = [f"clip-{i:06d}" for i in range(3365, 3832)]

# Thresholds to determine if an object is in hand
IOU_THRESHOLD = 0.1       # IoU threshold between hand and object
DISTANCE_THRESHOLD = 0.01  # Distance threshold between hand and object (in meters)
VELOCITY_THRESHOLD = 0.01  # Object velocity threshold (in m/s)

# RLE parameters for masks
RLE_CONVERSION_FORMAT = "coco"  # Format for RLE conversion (e.g., "coco", "frumpy")

# Visualization parameters
BLUE_MASK_COLOR = [0, 0, 255]
MASK_ALPHA = 0.6

# Saved model path
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "mask_rcnn_final_visor.pth")