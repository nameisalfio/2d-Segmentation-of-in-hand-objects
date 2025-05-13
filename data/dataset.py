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
            cache_file (string): Percorso al file .npy contenente la lista di dizionari.
            transform (callable, optional): Trasformazione opzionale da applicare
                                             a immagine e target.
        """
        self.transform = transform
        try:
            self.samples = np.load(cache_file, allow_pickle=True)
            print(f"Caricati {len(self.samples)} campioni grezzi dal file {cache_file}")
        except FileNotFoundError:
            print(f"ERRORE: File cache {cache_file} non trovato.")
            self.samples = []
            return # Esce se il file non esiste

        if not isinstance(self.samples, (np.ndarray, list)):
             print(f"ERRORE: Il contenuto di {cache_file} non è una lista o un array NumPy.")
             self.samples = []
             return

        valid_samples = []
        invalid_count = 0
        required_keys = {"image_id", "file_name", "image_path", "individual_masks",
                         "boxes", "labels", "object_ids", "image_shape"}

        for i, sample in enumerate(self.samples):
            if not isinstance(sample, dict):
                print(f"Campione {i} non è un dizionario, scartato.")
                invalid_count += 1
                continue

            # Controllo presenza chiavi essenziali
            missing_keys = required_keys - set(sample.keys())
            if missing_keys:
                # Permetti che 'mask' (overall) sia opzionale
                if missing_keys != {'mask'}:
                     print(f"Campione {i} (ID: {sample.get('image_id', 'N/A')}) mancano chiavi: {missing_keys}, scartato.")
                     invalid_count += 1
                     continue

            if self._is_valid_sample(sample, i):
                valid_samples.append(sample)
            else:
                invalid_count += 1

        if invalid_count > 0:
            print(f"Rimossi {invalid_count} campioni non validi o con errori.")

        self.samples = valid_samples
        print(f"Numero finale di campioni validi: {len(self.samples)}")

    def _is_valid_sample(self, sample, index):
        """Verifica la validità di un singolo campione."""
        image_id = sample.get("image_id", f"indice_{index}") # Usa l'indice se l'ID manca

        # 1. Verifica esistenza immagine
        if "image_path" not in sample or not os.path.exists(sample["image_path"]):
            print(f"ID {image_id}: Percorso immagine mancante o non esistente ({sample.get('image_path', 'N/A')})")
            return False

        # 2. Verifica coerenza numero istanze
        num_boxes = len(sample.get("boxes", []))
        num_labels = len(sample.get("labels", []))
        num_masks = len(sample.get("individual_masks", []))
        num_obj_ids = len(sample.get("object_ids", []))

        if not (num_boxes == num_labels == num_masks == num_obj_ids):
            print(f"ID {image_id}: Incoerenza nel numero di istanze: "
                  f"boxes={num_boxes}, labels={num_labels}, masks={num_masks}, obj_ids={num_obj_ids}")
            return False
        num_instances = num_boxes # Numero di oggetti rilevati

        # 3. Verifica validità Bounding Boxes (se presenti)
        if num_instances > 0:
            boxes = sample["boxes"]
            if not isinstance(boxes, (list, np.ndarray)):
                 print(f"ID {image_id}: 'boxes' non è una lista o ndarray.")
                 return False
            for i, box in enumerate(boxes):
                if len(box) != 4:
                    print(f"ID {image_id}: Box {i} non ha 4 coordinate.")
                    return False
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    print(f"ID {image_id}: Box {i} ha coordinate non valide ({box}).")
                    return False

        # 4. Verifica validità Maschere Individuali (se presenti)
        if num_instances > 0:
            masks = sample["individual_masks"]
            if not isinstance(masks, list):
                 print(f"ID {image_id}: 'individual_masks' non è una lista.")
                 return False
            for i, mask in enumerate(masks):
                if not isinstance(mask, np.ndarray):
                    print(f"ID {image_id}: Maschera individuale {i} non è un ndarray.")
                    return False
                if mask.ndim != 2:
                     print(f"ID {image_id}: Maschera individuale {i} non ha 2 dimensioni (shape={mask.shape}).")
                     return False
                 # Potremmo aggiungere un controllo sulle dimensioni rispetto a image_shape,
                 # ma lo gestiamo in __getitem__ con resize se necessario.

        # 5. Verifica image_shape
        if "image_shape" in sample:
            shape = sample["image_shape"]
            if not isinstance(shape, (tuple, list)) or len(shape) != 2:
                print(f"ID {image_id}: 'image_shape' non valido ({shape}).")
                return False
            if not all(isinstance(dim, int) and dim > 0 for dim in shape):
                 print(f"ID {image_id}: Dimensioni in 'image_shape' non valide ({shape}).")
                 return False
        else:
             print(f"ID {image_id}: Chiave 'image_shape' mancante.")
             return False # Consideriamo image_shape essenziale

        # 6. Verifica tipi Labels e Object IDs (se presenti)
        if num_instances > 0:
            if not all(isinstance(l, (int, np.integer)) for l in sample["labels"]):
                 print(f"ID {image_id}: Non tutte le 'labels' sono interi.")
                 return False
            if not all(isinstance(o, (int, np.integer)) for o in sample["object_ids"]):
                 print(f"ID {image_id}: Non tutti gli 'object_ids' sono interi.")
                 return False

        return True # Il campione è valido

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Prendi il dizionario del campione
        sample = self.samples[idx]

        # Carica l'immagine
        image_path = sample["image_path"]
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise IOError(f"cv2.imread ha restituito None per {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise RuntimeError(f"Errore durante il caricamento dell'immagine {image_path} (indice {idx}): {e}")

        # Ottieni le dimensioni reali dell'immagine caricata
        img_h, img_w = image.shape[:2]

        # Estrai le informazioni dal dizionario
        image_id = sample.get("image_id", idx) # Usa l'indice se manca l'ID
        boxes = sample.get("boxes", [])
        
        # Ottieni le etichette e modificale per farle iniziare da 0
        original_labels = sample.get("labels", [])
        # Sposta tutti gli ID di classe di -1 per renderli 0-indexed
        adjusted_labels = [label - 1 for label in original_labels]
        
        object_ids = sample.get("object_ids", [])
        individual_masks = sample.get("individual_masks", []) # Lista di ndarray

        num_instances = len(boxes)

        # Converti in Tensori PyTorch
        image = F.to_tensor(image) # Converte in CxHxW e normalizza a [0, 1]

        # Boxes: Converti in tensore Float32
        # Assicurati che sia un tensore 2D anche se vuoto o con una sola box
        if num_instances > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32).reshape(num_instances, 4)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        # Labels: Converti in tensore Int64 (ora 0-indexed)
        labels = torch.tensor(adjusted_labels, dtype=torch.int64)

        # Object IDs: Converti in tensore Int64
        object_ids = torch.tensor(object_ids, dtype=torch.int64)

        # Maschere Individuali: Converti e impila
        masks_list = []
        if num_instances > 0:
            for mask_np in individual_masks:
                # Verifica e adatta le dimensioni se necessario
                # Le maschere nei file .npy dovrebbero avere H, W
                if mask_np.shape != (img_h, img_w):
                    # Usa le dimensioni dell'immagine CARICATA (img_h, img_w)
                    mask_np = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

                # Converti in tensore uint8 (modelli come MaskRCNN si aspettano 0 o 1, uint8 va bene)
                masks_list.append(torch.as_tensor(mask_np, dtype=torch.uint8))

            # Impila le maschere in un unico tensore (N, H, W)
            masks = torch.stack(masks_list)
        else:
            # Nessun oggetto, crea un tensore vuoto con le dimensioni corrette
            masks = torch.zeros((0, img_h, img_w), dtype=torch.uint8)

        # Crea il dizionario target richiesto da molti modelli torchvision
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels  # Ora le etichette sono 0-indexed
        target["masks"] = masks # Maschere individuali impilate
        target["image_id"] = torch.tensor([image_id], dtype=torch.int64) # Deve essere un tensore
        target["object_ids"] = object_ids

        # Applica le trasformazioni (se presenti)
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
            # Immagine: HWC -> C H W (dopo to_tensor)
            image = image.flip(-1) # Flip sull'ultima dimensione (larghezza)

            # Maschere: N H W
            if "masks" in target and target["masks"].numel() > 0: # Controlla se ci sono maschere
                target["masks"] = target["masks"].flip(-1) # Flip sull'ultima dimensione (larghezza)

            # Boxes: x1, y1, x2, y2
            if "boxes" in target and target["boxes"].numel() > 0: # Controlla se ci sono boxes
                boxes = target["boxes"]
                img_width = image.shape[-1] # Prendi la larghezza dall'immagine flippata
                # Scambia x1 e x2 e sottrai dalla larghezza
                boxes[:, [0, 2]] = img_width - boxes[:, [2, 0]]
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
    """
    Combina un batch di (image, target) in un formato adatto per i modelli.
    Non fa padding qui, ma semplicemente raggruppa.
    """
    return tuple(zip(*batch))

def create_data_loaders(batch_size=2, num_workers=2): 
    """
    Crea data loader per training, validation e test.

    Args:
        batch_size (int): Dimensione del batch.
        num_workers (int): Numero di worker per il caricamento dati.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
               test_loader può essere None se il file non esiste.
    """

    if not os.path.exists(TRAIN_CACHE_PATH):
        print(f"ATTENZIONE: File {TRAIN_CACHE_PATH} non trovato. Il dataloader di training sarà vuoto.")

    if not os.path.exists(VAL_CACHE_PATH):
        print(f"ATTENZIONE: File {VAL_CACHE_PATH} non trovato. Il dataloader di validazione sarà vuoto.")

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Crea i dataset
    train_dataset = ObjectsInHandDataset(TRAIN_CACHE_PATH, transform=get_transform(train=True))
    val_dataset = ObjectsInHandDataset(VAL_CACHE_PATH, transform=get_transform(train=False))

    test_dataset = None
    test_loader = None
    if os.path.exists(TEST_CACHE_PATH):
        print(f"Trovato file di test: {TEST_CACHE_PATH}")
        test_dataset = ObjectsInHandDataset(TEST_CACHE_PATH, transform=get_transform(train=False))
    else:
        print(f"File di test {TEST_CACHE_PATH} non trovato.")

    # Crea i data loader
    persistent_workers_flag = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle per il training
        collate_fn=collate_fn, # Usa la collate_fn standard
        num_workers=num_workers,
        persistent_workers=persistent_workers_flag if len(train_dataset) > 0 else False,
        pin_memory=True, # Utile se si usa la GPU
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, # Puoi usare un batch size maggiore per la validazione
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers_flag if len(val_dataset) > 0 else False,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    if test_dataset is not None and len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Per il test, batch size 1 è spesso preferito per valutare immagine per immagine
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=persistent_workers_flag, # Manteniamo la coerenza
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        print(f"Creato test_loader con {len(test_dataset)} campioni.")
    elif test_dataset is not None and len(test_dataset) == 0:
         print("Il dataset di test è vuoto dopo la validazione dei campioni.")
    else:
         print("Nessun test_loader creato.")


    # Stampa info riassuntive
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset) if test_dataset is not None else 0

    print(f"Data loader creati:")
    print(f"  Training:   {train_size} campioni, {len(train_loader)} batch")
    print(f"  Validation: {val_size} campioni, {len(val_loader)} batch")
    if test_loader:
        print(f"  Test:       {test_size} campioni, {len(test_loader)} batch")
    else:
        print(f"  Test:       N/A")

    return train_loader, val_loader, test_loader

def build_dataset_files(train=False, val=False, test=False, debug=False):
    """
    Esegue lo script di preprocessing per creare i file .npy se non esistono.
    """
    from data.preprocessing import process_visor_dataset
    print("Trovato modulo data.preprocessing.")

    # Controlla HOT3D_DATASET_PATH (necessario per il preprocessing)
    if 'HOT3D_DATASET_PATH' not in globals() or not os.path.isdir(HOT3D_DATASET_PATH):
         print(f"ERRORE: La variabile HOT3D_DATASET_PATH non è definita o non è una directory valida.")
         print("Impossibile eseguire il preprocessing.")
         return

    # Definisci i percorsi ai file JSON di annotazione
    train_json_file = os.path.join(HOT3D_DATASET_PATH, "annotations", "train.json")
    val_json_file = os.path.join(HOT3D_DATASET_PATH, "annotations", "val.json")
    test_json_file = os.path.join(HOT3D_DATASET_PATH, "annotations", "test.json")

    # Crea i file .npy se richiesti e mancanti
    if train and not os.path.exists(TRAIN_CACHE_PATH):
        print(f"Creazione dataset di training ({TRAIN_CACHE_PATH})...")
        if os.path.exists(train_json_file):
            try:
                process_visor_dataset(train_json_file, HOT3D_DATASET_PATH, TRAIN_CACHE_PATH, debug)
                print(f"File {TRAIN_CACHE_PATH} creato.")
            except Exception as e:
                 print(f"ERRORE durante la creazione di {TRAIN_CACHE_PATH}: {e}")
        else:
            print(f"ERRORE: File di annotazioni per training non trovato: {train_json_file}")

    if val and not os.path.exists(VAL_CACHE_PATH):
        print(f"Creazione dataset di validation ({VAL_CACHE_PATH})...")
        if os.path.exists(val_json_file):
             try:
                process_visor_dataset(val_json_file, HOT3D_DATASET_PATH, VAL_CACHE_PATH, debug)
                print(f"File {VAL_CACHE_PATH} creato.")
             except Exception as e:
                 print(f"ERRORE durante la creazione di {VAL_CACHE_PATH}: {e}")
        else:
            print(f"ERRORE: File di annotazioni per validation non trovato: {val_json_file}")

    if test and not os.path.exists(TEST_CACHE_PATH):
        print(f"Creazione dataset di test ({TEST_CACHE_PATH})...")
        if os.path.exists(test_json_file):
            try:
                process_visor_dataset(test_json_file, HOT3D_DATASET_PATH, TEST_CACHE_PATH, debug)
                print(f"File {TEST_CACHE_PATH} creato.")
            except Exception as e:
                 print(f"ERRORE durante la creazione di {TEST_CACHE_PATH}: {e}")
        else:
            print(f"INFO: File di annotazioni per test non trovato: {test_json_file}. Il file .npy di test non sarà creato.")

    print("Controllo file dataset completato.")

def main():
    parser = argparse.ArgumentParser(description='Gestione Dataset VISOR (Preprocessing e Dataloader)')
    parser.add_argument('--action', choices=['preprocess', 'dataloader', 'all'], default='all',
                        help="Azione: 'preprocess' (crea/controlla file .npy), "
                             "'dataloader' (crea solo i loader), 'all' (entrambi)")
    parser.add_argument('--dataset', choices=['train', 'val', 'test', 'all'], default='all',
                        help="Quali dataset processare/controllare durante 'preprocess': train, val, test o all")
    parser.add_argument('--debug', action='store_true',
                        help="Attiva output/salvataggi di debug durante il preprocessing (se supportato)")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, # Usa valore da config o default
                        help=f'Dimensione del batch per i dataloader (default: {BATCH_SIZE})')
    parser.add_argument('--num_workers', type=int, default=4, # Valore comune di default
                        help='Numero di workers per i dataloader (default: 4)')

    args = parser.parse_args()

    # Azione 1: Preprocessing (Creazione/Controllo file .npy)
    if args.action in ['preprocess', 'all']:
        print("\n--- ESECUZIONE PREPROCESSING / CONTROLLO FILE CACHE ---")
        process_train = args.dataset in ['train', 'all']
        process_val = args.dataset in ['val', 'all']
        process_test = args.dataset in ['test', 'all']
        build_dataset_files(process_train, process_val, process_test, args.debug)
        print("-------------------------------------------------------\n")


    # Azione 2: Creazione Dataloader
    if args.action in ['dataloader', 'all']:
        print("--- CREAZIONE DATALOADER ---")

        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        print("Dataloader creati con successo.")
        print("---------------------------\n")


if __name__ == "__main__":
    main()