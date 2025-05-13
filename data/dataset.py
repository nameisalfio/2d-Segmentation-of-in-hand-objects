import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as F
import sys
import random
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import * 


class ObjectsInHandDataset(torch.utils.data.Dataset):
    def __init__(self, cache_file, transform=None):
        """
        Args:
            cache_file (string): Path to the .npy file containing the list of dictionaries.
            transform (callable, optional): Optional transform to be applied
                                             to image and target.
        """
        self.transform = transform
        try:
            self.samples = np.load(cache_file, allow_pickle=True)
            print(f"Loaded {len(self.samples)} raw samples from {cache_file}")
        except FileNotFoundError:
            print(f"ERROR: Cache file {cache_file} not found.")
            self.samples = []
            return # Exit if the file does not exist

        if not isinstance(self.samples, (np.ndarray, list)):
             print(f"ERROR: Content of {cache_file} is not a list or NumPy array.")
             self.samples = []
             return

        valid_samples = []
        invalid_count = 0
        # Define essential keys that each sample dictionary must contain
        required_keys = {"image_id", "file_name", "image_path", "individual_masks",
                         "boxes", "labels", "object_ids", "image_shape"}

        for i, sample in enumerate(self.samples):
            if not isinstance(sample, dict):
                print(f"Sample {i} is not a dictionary, discarding.")
                invalid_count += 1
                continue

            # Check for essential keys
            missing_keys = required_keys - set(sample.keys())
            if missing_keys:
                # Allow 'mask' (overall combined mask) to be optional
                if missing_keys != {'mask'}:
                     print(f"Sample {i} (ID: {sample.get('image_id', 'N/A')}) missing keys: {missing_keys}, discarding.")
                     invalid_count += 1
                     continue

            if self._is_valid_sample(sample, i):
                valid_samples.append(sample)
            else:
                invalid_count += 1

        if invalid_count > 0:
            print(f"Removed {invalid_count} invalid samples or samples with errors.")

        self.samples = valid_samples
        print(f"Final number of valid samples: {len(self.samples)}")

    def _is_valid_sample(self, sample, index):
        """Checks the validity of a single sample."""
        image_id = sample.get("image_id", f"index_{index}") # Use index if ID is missing

        # 1. Check image existence
        if "image_path" not in sample or not os.path.exists(sample["image_path"]):
            print(f"ID {image_id}: Image path missing or non-existent ({sample.get('image_path', 'N/A')})")
            return False

        # 2. Check instance count consistency
        num_boxes = len(sample.get("boxes", []))
        num_labels = len(sample.get("labels", []))
        num_masks = len(sample.get("individual_masks", []))
        num_obj_ids = len(sample.get("object_ids", []))

        if not (num_boxes == num_labels == num_masks == num_obj_ids):
            print(f"ID {image_id}: Inconsistent number of instances: "
                  f"boxes={num_boxes}, labels={num_labels}, masks={num_masks}, obj_ids={num_obj_ids}")
            return False
        num_instances = num_boxes # Number of detected objects

        # 3. Check Bounding Box validity (if present)
        if num_instances > 0:
            boxes = sample["boxes"]
            if not isinstance(boxes, (list, np.ndarray)):
                 print(f"ID {image_id}: 'boxes' is not a list or ndarray.")
                 return False
            for i, box in enumerate(boxes):
                if len(box) != 4:
                    print(f"ID {image_id}: Box {i} does not have 4 coordinates.")
                    return False
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1: # Checks for valid box dimensions
                    print(f"ID {image_id}: Box {i} has invalid coordinates ({box}).")
                    return False

        # 4. Check Individual Mask validity (if present)
        if num_instances > 0:
            masks = sample["individual_masks"]
            if not isinstance(masks, list):
                 print(f"ID {image_id}: 'individual_masks' is not a list.")
                 return False
            for i, mask_item in enumerate(masks): # Renamed 'mask' to 'mask_item' to avoid conflict
                if not isinstance(mask_item, np.ndarray):
                    print(f"ID {image_id}: Individual mask {i} is not an ndarray.")
                    return False
                if mask_item.ndim != 2:
                     print(f"ID {image_id}: Individual mask {i} does not have 2 dimensions (shape={mask_item.shape}).")
                     return False
                 # We could add a check on dimensions against image_shape here,
                 # but it's handled in __getitem__ with resize if necessary.

        # 5. Check image_shape
        if "image_shape" in sample:
            shape = sample["image_shape"] # Expected (H, W)
            if not isinstance(shape, (tuple, list)) or len(shape) != 2:
                print(f"ID {image_id}: 'image_shape' is invalid ({shape}).")
                return False
            if not all(isinstance(dim, int) and dim > 0 for dim in shape):
                 print(f"ID {image_id}: Dimensions in 'image_shape' are invalid ({shape}).")
                 return False
        else:
             print(f"ID {image_id}: 'image_shape' key missing.")
             return False # We consider image_shape essential

        # 6. Check Labels and Object IDs types (if present)
        if num_instances > 0:
            if not all(isinstance(l, (int, np.integer)) for l in sample["labels"]):
                 print(f"ID {image_id}: Not all 'labels' are integers.")
                 return False
            if not all(isinstance(o, (int, np.integer)) for o in sample["object_ids"]):
                 print(f"ID {image_id}: Not all 'object_ids' are integers.")
                 return False

        return True # Sample is valid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx] # Get sample data dictionary

        image_path = sample["image_path"]
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise IOError(f"cv2.imread returned None for {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path} (index {idx}): {e}")

        img_h, img_w = image.shape[:2] # Get actual dimensions of the loaded image

        image_id = sample.get("image_id", idx) # Use index if ID is missing
        boxes = sample.get("boxes", [])
        
        original_labels = sample.get("labels", [])
        adjusted_labels = [label for label in original_labels] # No adjustment if labels are already for model

        object_ids = sample.get("object_ids", [])
        individual_masks = sample.get("individual_masks", []) # List of ndarrays

        num_instances = len(boxes)

        image = F.to_tensor(image) # Converts to CxHxW and scales to [0, 1]

        # Boxes: Convert to Float32 tensor
        # Ensure it's a 2D tensor even if empty or with a single box
        if num_instances > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32).reshape(num_instances, 4)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        # Labels: Convert to Int64 tensor
        labels = torch.tensor(adjusted_labels, dtype=torch.int64)

        # Object IDs: Convert to Int64 tensor
        object_ids = torch.tensor(object_ids, dtype=torch.int64)

        # Individual Masks: Convert and stack
        masks_list = []
        if num_instances > 0:
            for mask_np in individual_masks:
                # Masks in .npy files should have H, W dimensions
                if mask_np.shape != (img_h, img_w):
                    # Use dimensions of the LOADED image (img_h, img_w) for resizing
                    mask_np = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                # Convert to uint8 tensor (models like MaskRCNN expect 0 or 1, uint8 is fine)
                masks_list.append(torch.as_tensor(mask_np, dtype=torch.uint8))
            # Stack masks into a single tensor (N, H, W)
            masks = torch.stack(masks_list)
        else:
            # No objects, create an empty tensor with correct dimensions
            masks = torch.zeros((0, img_h, img_w), dtype=torch.uint8)

        # Create the target dictionary required by many torchvision models
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks # Stacked individual masks
        target["image_id"] = torch.tensor([image_id], dtype=torch.int64) # Must be a tensor
        target["object_ids"] = object_ids
        # Optional: Add area and iscrowd if your model uses them
        # target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # target["iscrowd"] = torch.zeros((num_instances,), dtype=torch.int64)


        # Apply transformations (if present)
        if self.transform:
            image, target = self.transform(image, target)
    
        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            # Image is C x H x W (after to_tensor)
            image = image.flip(-1) # Flip on the last dimension (width)

            # Masks are N x H x W
            if "masks" in target and target["masks"].numel() > 0: # Check if there are masks
                target["masks"] = target["masks"].flip(-1) # Flip on the last dimension (width)

            # Boxes are (x1, y1, x2, y2)
            if "boxes" in target and target["boxes"].numel() > 0: # Check if there are boxes
                boxes = target["boxes"]
                img_width = image.shape[-1] # Get width from the (already flipped) image tensor
                # Swap x1 and x2, then subtract from width
                boxes[:, [0, 2]] = img_width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target

class Normalize: # Standard normalization for ImageNet-pretrained models
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

def get_transform(train):
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    # Normalization is typically applied for both train and val/test
    transforms.append(Normalize()) # Assuming image is already a tensor in [0,1] range

    return Compose(transforms)

def collate_fn(batch):
    """
    Custom collate function for object detection.
    It simply returns a tuple of images and a tuple of targets.
    """
    return tuple(zip(*batch))

def create_data_loaders(batch_size=2, num_workers=2): 
    """
    Creates data loaders for training, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
               test_loader can be None if its cache file does not exist or is empty.
    """

    if not os.path.exists(TRAIN_CACHE_PATH):
        print(f"WARNING: Training cache file {TRAIN_CACHE_PATH} not found. Training dataloader will be empty or fail.")

    if not os.path.exists(VAL_CACHE_PATH):
        print(f"WARNING: Validation cache file {VAL_CACHE_PATH} not found. Validation dataloader will be empty or fail.")

    def worker_init_fn(worker_id): # For reproducibility with multiple workers
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Create datasets
    train_dataset = ObjectsInHandDataset(TRAIN_CACHE_PATH, transform=get_transform(train=True))
    val_dataset = ObjectsInHandDataset(VAL_CACHE_PATH, transform=get_transform(train=False))

    test_dataset = None
    test_loader = None
    if os.path.exists(TEST_CACHE_PATH):
        print(f"Found test cache file: {TEST_CACHE_PATH}")
        test_dataset = ObjectsInHandDataset(TEST_CACHE_PATH, transform=get_transform(train=False))
        if len(test_dataset) == 0:
            print("Test dataset is empty after sample validation. Test loader will not be created.")
            test_dataset = None # Ensure test_loader is not created for an empty dataset
    else:
        print(f"Test cache file {TEST_CACHE_PATH} not found.")

    # Create data loaders
    # persistent_workers useful if num_workers > 0 to keep workers alive between epochs
    persistent_workers_flag = num_workers > 0 

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle for training
        collate_fn=collate_fn, # Use the custom collate_fn
        num_workers=num_workers,
        persistent_workers=persistent_workers_flag if len(train_dataset) > 0 else False,
        pin_memory=True, # Speeds up CPU to GPU transfer
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, # Can use a larger batch size for validation if memory allows
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers_flag if len(val_dataset) > 0 else False,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    if test_dataset is not None: # Check if test_dataset was successfully created and is not empty
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # For testing, batch size 1 is often preferred for image-by-image evaluation
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=persistent_workers_flag, 
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        print(f"Created test_loader with {len(test_dataset)} samples.")
    else:
         print("No test_loader created (test dataset not found, empty, or error during init).")


    # Print summary info
    train_size = len(train_dataset) if train_dataset is not None else 0
    val_size = len(val_dataset) if val_dataset is not None else 0
    test_size = len(test_dataset) if test_dataset is not None else 0

    print(f"Data loaders created:")
    print(f"  Training:   {train_size} samples, {len(train_loader) if train_loader else 0} batches")
    print(f"  Validation: {val_size} samples, {len(val_loader) if val_loader else 0} batches")
    if test_loader:
        print(f"  Test:       {test_size} samples, {len(test_loader)} batches")
    else:
        print(f"  Test:       N/A")

    return train_loader, val_loader, test_loader

def build_dataset_files(train=False, val=False, test=False, debug=False):
    """
    Runs the preprocessing script to create .npy cache files if they don't exist.
    """
    try:
        from data.preprocessing import process_visor_dataset # Assumes this function exists
        print("Found data.preprocessing module.")
    except ImportError:
        print("ERROR: Could not import 'process_visor_dataset' from 'data.preprocessing'.")
        print("Ensure 'data/preprocessing.py' exists and is correctly structured.")
        return

    # Check HOT3D_DATASET_PATH (required for preprocessing)
    if 'HOT3D_DATASET_PATH' not in globals() or not HOT3D_DATASET_PATH or not os.path.isdir(HOT3D_DATASET_PATH):
         print(f"ERROR: The HOT3D_DATASET_PATH variable is not defined, empty, or not a valid directory.")
         print("Cannot run preprocessing.")
         return

    # Define paths to annotation JSON files
    train_json_file = os.path.join(HOT3D_DATASET_PATH, "annotations", "train.json")
    val_json_file = os.path.join(HOT3D_DATASET_PATH, "annotations", "val.json")
    test_json_file = os.path.join(HOT3D_DATASET_PATH, "annotations", "test.json")

    # Create .npy files if requested and missing
    if train and not os.path.exists(TRAIN_CACHE_PATH):
        print(f"Creating training dataset cache ({TRAIN_CACHE_PATH})...")
        if os.path.exists(train_json_file):
            try:
                process_visor_dataset(train_json_file, HOT3D_DATASET_PATH, TRAIN_CACHE_PATH, debug)
                print(f"File {TRAIN_CACHE_PATH} created.")
            except Exception as e:
                 print(f"ERROR during creation of {TRAIN_CACHE_PATH}: {e}")
        else:
            print(f"ERROR: Annotation file for training not found: {train_json_file}")

    if val and not os.path.exists(VAL_CACHE_PATH):
        print(f"Creating validation dataset cache ({VAL_CACHE_PATH})...")
        if os.path.exists(val_json_file):
             try:
                process_visor_dataset(val_json_file, HOT3D_DATASET_PATH, VAL_CACHE_PATH, debug)
                print(f"File {VAL_CACHE_PATH} created.")
             except Exception as e:
                 print(f"ERROR during creation of {VAL_CACHE_PATH}: {e}")
        else:
            print(f"ERROR: Annotation file for validation not found: {val_json_file}")

    if test and not os.path.exists(TEST_CACHE_PATH):
        print(f"Creating test dataset cache ({TEST_CACHE_PATH})...")
        if os.path.exists(test_json_file):
            try:
                process_visor_dataset(test_json_file, HOT3D_DATASET_PATH, TEST_CACHE_PATH, debug)
                print(f"File {TEST_CACHE_PATH} created.")
            except Exception as e:
                 print(f"ERROR during creation of {TEST_CACHE_PATH}: {e}")
        else:
            print(f"INFO: Annotation file for test not found: {test_json_file}. Test .npy cache will not be created.")

    print("Dataset file check completed.")

def main():
    parser = argparse.ArgumentParser(description='VISOR Dataset Management (Preprocessing and Dataloaders)')
    parser.add_argument('--action', choices=['preprocess', 'dataloader', 'all'], default='all',
                        help="Action: 'preprocess' (creates/checks .npy files), "
                             "'dataloader' (creates only loaders), 'all' (both)")
    parser.add_argument('--dataset', choices=['train', 'val', 'test', 'all'], default='all',
                        help="Which datasets to process/check during 'preprocess': train, val, test, or all")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug output/saves during preprocessing (if supported by preprocessing script)")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE if 'BATCH_SIZE' in globals() else 2,
                        help=f'Batch size for dataloaders (default: from config or 2)')
    parser.add_argument('--num_workers', type=int, default=4, # Common default
                        help='Number of workers for dataloaders (default: 4)')

    args = parser.parse_args()

    # Action 1: Preprocessing (Create/Check .npy files)
    if args.action in ['preprocess', 'all']:
        print("\n--- EXECUTING PREPROCESSING / CACHE FILE CHECK ---")
        process_train = args.dataset in ['train', 'all']
        process_val = args.dataset in ['val', 'all']
        process_test = args.dataset in ['test', 'all']
        build_dataset_files(process_train, process_val, process_test, args.debug)
        print("-------------------------------------------------------\n")


    # Action 2: Dataloader Creation
    if args.action in ['dataloader', 'all']:
        print("--- CREATING DATALOADERS ---")

        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        print("---------------------------\n")


if __name__ == "__main__":
    main()