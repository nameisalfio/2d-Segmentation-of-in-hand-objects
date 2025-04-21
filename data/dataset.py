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
from data.preprocessing import process_dataset
from data.utils import clean_directory

class ObjectsInHandDataset(torch.utils.data.Dataset):
    def __init__(self, cache_file, transform=None):
        self.transform = transform
        self.samples = np.load(cache_file, allow_pickle=True)
        print(f"Caricati {len(self.samples)} campioni dal file {cache_file}")
        
        valid_samples = []
        invalid_count = 0
        for i, sample in enumerate(self.samples):
            if self._is_valid_sample(sample):
                valid_samples.append(sample)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"Rimossi {invalid_count} campioni non validi")
            self.samples = np.array(valid_samples, dtype=object)
    
    def _is_valid_sample(self, sample):
        if "image_path" not in sample or not os.path.exists(sample["image_path"]):
            return False
        
        if "boxes" in sample and len(sample["boxes"]) > 0:
            if any(len(box) != 4 for box in sample["boxes"]):
                return False
            
            for box in sample["boxes"]:
                if box[2] <= box[0] or box[3] <= box[1]:
                    return False
        
        return True
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image_path = sample["image_path"]
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Impossibile leggere l'immagine: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = sample.get("mask", np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))
        
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        image = F.to_tensor(image)
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        
        boxes = sample.get("boxes", [])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        
        labels = sample.get("labels", [])
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
        
        if "individual_masks" in sample and sample["individual_masks"]:
            individual_masks = []
            for m in sample["individual_masks"]:
                if m.shape[:2] != image.shape[1:]:
                    m = cv2.resize(m, (image.shape[2], image.shape[1]), 
                                  interpolation=cv2.INTER_NEAREST)
                individual_masks.append(torch.as_tensor(m, dtype=torch.uint8))
            masks = torch.stack(individual_masks) if individual_masks else torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
        else:
            masks = mask.unsqueeze(0)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx], dtype=torch.int64)
        }
        
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
            image = image.flip(-1)
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = image.shape[-1] - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target

class Normalize:
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
    
    transforms.append(Normalize())
    
    return Compose(transforms)
    
def collate_fn(batch):
    return tuple(zip(*batch))

def create_data_loaders(batch_size=BATCH_SIZE):
    """
    Crea data loader per training, validation e test.
    
    Args:
        batch_size: Dimensione del batch
        
    Returns:
        train_loader, val_loader, test_loader: Data loader per training, validation e test
    """
    if not os.path.exists(TRAIN_CACHE_PATH):
        raise FileNotFoundError(f"File {TRAIN_CACHE_PATH} non trovato. Esegui prima preprocessing.py")

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dataset = ObjectsInHandDataset(TRAIN_CACHE_PATH, transform=get_transform(train=True))
    
    if os.path.exists(VAL_CACHE_PATH):
        val_dataset = ObjectsInHandDataset(VAL_CACHE_PATH, transform=get_transform(train=False))
    else:
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        indices = torch.randperm(len(train_dataset)).tolist()
        train_dataset = Subset(train_dataset, indices[:train_size])
        val_dataset = Subset(train_dataset, indices[train_size:])
        
        print(f"Creati subset per training ({train_size} campioni) e validation ({val_size} campioni)")

    if os.path.exists(TEST_CACHE_PATH):
        test_dataset = ObjectsInHandDataset(TEST_CACHE_PATH, transform=get_transform(train=False))
        print(f"Caricato file di test: {TEST_CACHE_PATH}")
    else:
        test_dataset = []

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    print(f"Creati data loader: {len(train_dataset)} campioni per training, "
          f"{len(val_dataset)} per validation, {len(test_dataset)} per test")
    
    return train_loader, val_loader, test_loader

def build_dataset_files(train=True, test=True, max_clips='all', debug=False, max_frames_per_clip=None):
    """
    Crea i file di dataset (train, val e test) se non esistono già.
    
    Args:
        train: Se True, processa il dataset di training
        test: Se True, processa il dataset di test
        max_clips: Numero massimo di clip da processare o 'all' per tutti
        debug: Se True, salva immagini di debug
        max_frames_per_clip: Numero massimo di frame da selezionare per ogni clip
    """
    if train and (not os.path.exists(TRAIN_CACHE_PATH) or not os.path.exists(VAL_CACHE_PATH)):
        print("Creazione dataset di training...")
        process_dataset('train', max_clips, USE_CAMERAS, debug, max_frames_per_clip)
    
    if test and not os.path.exists(TEST_CACHE_PATH):
        print("Creazione dataset di test...")
        process_dataset('test', max_clips, USE_CAMERAS, debug, max_frames_per_clip)
    
    print("Dataset pronti per l'uso.") 
    
def main():
    parser = argparse.ArgumentParser(description='Preprocessing e creazione dataloader HOT3D')
    parser.add_argument('--action', choices=['preprocess', 'dataloader', 'all', 'clean'], default='all',
                        help='Azione da eseguire: preprocess, dataloader, all o clean')
    parser.add_argument('--dataset_type', choices=['train', 'test', 'both'], default='both',
                        help='Tipo di dataset da processare: train, test o both')
    parser.add_argument('--max_clips', default='all', 
                        help='Numero massimo di clip da processare (o "all" per tutti)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Numero massimo di frame da selezionare randomicamente per ogni clip')
    parser.add_argument('--debug', action='store_true', help='Salva immagini di debug')
    parser.add_argument('--cameras', nargs='+', default=USE_CAMERAS, help='ID telecamere da processare')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Dimensione del batch')
    
    args = parser.parse_args()
    
    if args.action == 'clean':
        if os.path.exists(DATASET_CACHE_DIR):
            print(f"Pulizia directory: {DATASET_CACHE_DIR}")
            clean_directory(DATASET_CACHE_DIR)
            print("Directory pulita.")
        return
    
    if args.action in ['preprocess', 'all']:
        process_train = args.dataset_type in ['train', 'both']
        process_test = args.dataset_type in ['test', 'both']
        build_dataset_files(process_train, process_test, args.max_clips, args.debug, args.max_frames)
    
    if args.action in ['dataloader', 'all']:
        train_loader, val_loader, test_loader = create_data_loaders(batch_size=args.batch_size)
        print("Dataset e dataloader creati con successo.")
        print(f"File di dataset: {TRAIN_CACHE_PATH}, {VAL_CACHE_PATH}, {TEST_CACHE_PATH}")
        print(f"Numero di batch nel train_loader: {len(train_loader)}")
        print(f"Numero di batch nel val_loader: {len(val_loader)}")
        print(f"Numero di batch nel test_loader: {len(test_loader)}")

if __name__ == "__main__":
    main()