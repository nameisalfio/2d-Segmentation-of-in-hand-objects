# Preprocessing HOT3D per Mask R-CNN

Questo progetto preprocessa il dataset HOT3D per creare un dataset di segmentazione di oggetti in mano. Il sistema estrae maschere binarie di oggetti tenuti in mano in immagini egocentriche.

## Struttura del progetto

```
mask_rcnn_project/
│
├── config.py                  # Configurazione globale
├── data/
│   ├── __init__.py
│   ├── utils.py              # Funzioni di utilità 
│   ├── preprocessing.py      # Funzioni di preprocessing
│   └── dataset.py            # Creazione dataset e dataloader (punto di ingresso principale)
├── dataset_cache/            # Directory per i file di cache
│   ├── train_dataset.npy     # Dataset di training
│   ├── val_dataset.npy       # Dataset di validazione
│   └── test_dataset.npy      # Dataset di test
└── debug_output/             # Output di debug
```

## Punto di ingresso principale

Il file principale è `data/dataset.py` che coordina il processo completo. Per eseguire tutto il pipeline, usa:

```bash
python data/dataset.py --action all --dataset_type both --max_clips all
```

Questo comando eseguirà il preprocessing di tutti i clip di training e test, e creerà i dataloader per l'uso con PyTorch.

## Parametri

```
--action {preprocess,dataloader,all,clean}
                      Azione da eseguire
--dataset_type {train,test,both}
                      Tipo di dataset da processare
--max_clips MAX_CLIPS
                      Numero massimo di clip da processare (o "all" per tutti)
--debug               Salva immagini di debug
--cameras CAMERAS [CAMERAS ...]
                      ID telecamere da processare
--batch_size BATCH_SIZE
                      Dimensione del batch
```

## Esempi di utilizzo

### Preprocessing di un sottoinsieme di clip di training

```bash
python data/dataset.py --action preprocess --dataset_type train --max_clips 5 --debug
```

### Creazione dei dataloader usando dati esistenti

```bash
python data/dataset.py --action dataloader --batch_size 4
```

### Pulizia della directory di cache

```bash
python data/dataset.py --action clean
```

## Criteri per "Oggetti in mano"

Un oggetto è considerato "in mano" se soddisfa almeno uno dei seguenti criteri:

1. La maschera dell'oggetto si sovrappone alla maschera della mano
2. La distanza 3D tra l'oggetto e la mano è inferiore a 1 cm (0.01 m)
3. L'IoU (Intersection over Union) tra il bounding box dell'oggetto e quello della mano supera la soglia definita in config.py

## Output

Il sistema genera tre file principali:

1. `train_dataset.npy`: Dati di training (80% del dataset di training)
2. `val_dataset.npy`: Dati di validazione (20% del dataset di training)
3. `test_dataset.npy`: Dati di test

Ogni file contiene una lista di dizionari con la seguente struttura:

```python
{
    "frame_id": "000042",               # ID del frame
    "camera_id": "214-1",               # ID della telecamera
    "image_path": "/path/to/image.jpg", # Percorso all'immagine
    "mask": np.array(...),              # Maschera binaria degli oggetti in mano
    "image_shape": (480, 640),          # Dimensioni dell'immagine
    "clip_name": "clip-001849"          # Nome del clip
}
```

I dataloader PyTorch restituiscono coppie `(image, mask)` dove:
- `image`: Tensore PyTorch dell'immagine normalizzata (3, H, W)
- `mask`: Tensore PyTorch della maschera binaria (H, W)