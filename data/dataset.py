import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as F
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from data.preprocessing import process_dataset
from data.utils import clean_directory

class ObjectsInHandDataset(Dataset):
    """Dataset PyTorch per la segmentazione degli oggetti in mano."""
    
    def __init__(self, data_path, transform=None):
        """
        Inizializza il dataset.
        
        Args:
            data_path: Percorso al file .npy contenente i dati
            transform: Trasformazioni da applicare
        """
        self.transform = transform
        self.samples = []
        
        # Carica i dati
        if os.path.exists(data_path):
            try:
                data = np.load(data_path, allow_pickle=True)
                print(f"Caricati {len(data)} campioni dal file {data_path}")
                self.samples = data
            except Exception as e:
                print(f"Errore nel caricare {data_path}: {str(e)}")
        else:
            print(f"File {data_path} non trovato!")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Carica l'immagine
        image_path = sample["image_path"]
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Impossibile leggere l'immagine: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Errore nel caricare l'immagine {image_path}: {str(e)}")
            # Fallback: immagine vuota
            height, width = sample["image_shape"]
            image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Ottieni la maschera
        mask = sample["mask"]
        
        # Assicurati che la maschera abbia le dimensioni corrette
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        # Converti l'immagine e la maschera in tensori
        image = F.to_tensor(image)
        mask = torch.as_tensor(mask, dtype=torch.long)  # usa long per maschere di classe
        
        # Applica trasformazioni se necessario
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask  # Ritorna solo l'immagine e la maschera

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob
    
    def __call__(self, image, mask):
        if torch.rand(1) < self.prob:
            image = image.flip(-1)
            mask = mask.flip(-1)
        return image, mask

class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, mask):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask

def get_transform(train):
    """
    Ottiene le trasformazioni appropriate per l'addestramento o la validazione.
    
    Args:
        train: Se True, include trasformazioni di data augmentation
        
    Returns:
        Funzione di trasformazione
    """
    transforms = []
    
    # Aggiungi data augmentation solo per il training
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    
    # Normalizzazione
    transforms.append(Normalize())
    
    return Compose(transforms)

def collate_fn(batch):
    """
    Funzione collate personalizzata per gestire batch con oggetti variabili.
    
    Args:
        batch: Batch di dati da collate
        
    Returns:
        Tuple di (images, masks)
    """
    return tuple(zip(*batch))

def create_data_loaders(batch_size=BATCH_SIZE):
    """
    Crea i data loader per training, validation e test.
    
    Args:
        batch_size: Dimensione del batch
        
    Returns:
        Tuple di (train_loader, val_loader, test_loader)
    """
    # Verifica che i file di cache esistano
    if not os.path.exists(TRAIN_CACHE_PATH):
        raise FileNotFoundError(f"File {TRAIN_CACHE_PATH} non trovato. Esegui prima preprocessing.py")
    
    # Crea i dataset direttamente dai file salvati
    train_dataset = ObjectsInHandDataset(TRAIN_CACHE_PATH, transform=get_transform(train=True))
    
    # Usa il file di validazione se esiste, altrimenti crea un subset
    if os.path.exists(VAL_CACHE_PATH):
        val_dataset = ObjectsInHandDataset(VAL_CACHE_PATH, transform=get_transform(train=False))
        print(f"Caricato file di validazione: {VAL_CACHE_PATH}")
    else:
        # Dividi il dataset in train e validation
        train_size = int(0.8 * len(train_dataset))
        
        indices = list(range(len(train_dataset)))
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Crea subset
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(ObjectsInHandDataset(TRAIN_CACHE_PATH, 
                                              transform=get_transform(train=False)), val_indices)
        print("Creati subset per training e validation dal dataset principale")
    
    # Crea test dataset se esiste
    if os.path.exists(TEST_CACHE_PATH):
        test_dataset = ObjectsInHandDataset(TEST_CACHE_PATH, transform=get_transform(train=False))
        print(f"Caricato file di test: {TEST_CACHE_PATH}")
    else:
        test_dataset = val_dataset  # Usa validation come test se test non esiste
        print("File di test non trovato, utilizzo validation come test")
        
    # Crea data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Per il test è meglio usare batch_size=1
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    print(f"Creati data loader: {len(train_dataset)} campioni per training, "
          f"{len(val_dataset)} per validation, {len(test_dataset)} per test")
    
    return train_loader, val_loader, test_loader

def build_dataset_files(train=True, test=True, max_clips='all', debug=False):
    """
    Crea i file di dataset (train, val e test) se non esistono già.
    
    Args:
        train: Se True, processa il dataset di training
        test: Se True, processa il dataset di test
        max_clips: Numero massimo di clip da processare o 'all' per tutti
        debug: Se True, salva immagini di debug
    """
    if train and (not os.path.exists(TRAIN_CACHE_PATH) or not os.path.exists(VAL_CACHE_PATH)):
        print("Creazione dataset di training...")
        process_dataset('train', max_clips, USE_CAMERAS, debug)
    
    if test and not os.path.exists(TEST_CACHE_PATH):
        print("Creazione dataset di test...")
        process_dataset('test', max_clips, USE_CAMERAS, debug)
    
    print("Dataset pronti per l'uso.")

def main():
    parser = argparse.ArgumentParser(description='Preprocessing e creazione dataloader HOT3D')
    parser.add_argument('--action', choices=['preprocess', 'dataloader', 'all', 'clean'], default='all',
                        help='Azione da eseguire: preprocess, dataloader, all o clean')
    parser.add_argument('--dataset_type', choices=['train', 'test', 'both'], default='both',
                        help='Tipo di dataset da processare: train, test o both')
    parser.add_argument('--max_clips', default='all', 
                        help='Numero massimo di clip da processare (o "all" per tutti)')
    parser.add_argument('--debug', action='store_true', help='Salva immagini di debug')
    parser.add_argument('--cameras', nargs='+', default=USE_CAMERAS, help='ID telecamere da processare')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Dimensione del batch')
    
    args = parser.parse_args()
    
    # Pulisci la directory di dataset_cache
    if args.action == 'clean':
        if os.path.exists(DATASET_CACHE_DIR):
            print(f"Pulizia directory: {DATASET_CACHE_DIR}")
            clean_directory(DATASET_CACHE_DIR)
            print("Directory pulita.")
        return
    
    # Esegui il preprocessing
    if args.action in ['preprocess', 'all']:
        process_train = args.dataset_type in ['train', 'both']
        process_test = args.dataset_type in ['test', 'both']
        build_dataset_files(process_train, process_test, args.max_clips, args.debug)
    
    # Crea i dataset/dataloader
    if args.action in ['dataloader', 'all']:
        train_loader, val_loader, test_loader = create_data_loaders(batch_size=args.batch_size)
        print("Dataset e dataloader creati con successo.")
        print(f"File di dataset: {TRAIN_CACHE_PATH}, {VAL_CACHE_PATH}, {TEST_CACHE_PATH}")
        print(f"Numero di batch nel train_loader: {len(train_loader)}")
        print(f"Numero di batch nel val_loader: {len(val_loader)}")
        print(f"Numero di batch nel test_loader: {len(test_loader)}")

if __name__ == "__main__":
    main()