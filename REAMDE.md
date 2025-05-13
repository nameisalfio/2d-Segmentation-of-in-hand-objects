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
python data/dataset.py --action preprocess --dataset train --debug
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

1. `train_dataset.npy`: Dati di training 
2. `val_dataset.npy`: Dati di validazione 
3. `test_dataset.npy`: Dati di test

Ogni file contiene una lista di dizionari con la seguente struttura:

```python
{
    "image_id": 42743,                                     # ID univoco dell'immagine
    "file_name": "camera_42743.jpg",                       # Nome del file immagine
    "image_path": "/storage/aspoto/visor_egohos_synth/images/camera_42743.jpg",  # Percorso assoluto
    "mask": np.ndarray shape=(720, 1280), dtype=np.uint8,  # Maschera binaria complessiva
    "individual_masks": [np.ndarray],                      # Lista di maschere individuali (una per oggetto)
    "boxes": [[345.0, 555.0, 501.0, 720.0]],                # Bounding boxes [x1, y1, x2, y2]
    "labels": [2],                                         # Classi associate agli oggetti
    "object_ids": [194353],                                # ID univoci degli oggetti
    "image_shape": (720, 1280)                             # Dimensioni dell'immagine (altezza, larghezza)
}
```

I dataloader PyTorch restituiscono .. specificare come sono organizzati 

## Main

### Addestramento
```bash
python main.py train --batch_size 2 --epochs 20
```

### Valutazione
```bash
python main.py evaluate --model saved_models/mask_rcnn_final.pth --dataset_type val
```

### Inferenza
```bash
python main.py inference --model saved_models/mask_rcnn_final.pth 
```

### Visualizzazione di campioni
```bash
python main.py visualize --dataset train --num_samples 10
```