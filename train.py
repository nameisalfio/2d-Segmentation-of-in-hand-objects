import os
import sys
import random
import numpy as np
import torch
import datetime
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Imposta i seed per la riproducibilità
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Importa i moduli del progetto
from config import (MODELS_DIR, VISUALIZATIONS_DIR, TENSORBOARD_DIR, 
                   RANDOM_SEED, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, 
                   NUM_CLASSES, MODEL_SAVE_PATH)
                   
from data.preprocessing import prepare_datasets
from data.utils import collate_fn
from models.mask_rcnn import MaskRCNNModel
from utils.visualization import visualize_dataset_samples, visualize_training_history, visualize_prediction

def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, model_save_path=MODEL_SAVE_PATH):
    """
    Addestra il modello Mask R-CNN per la segmentazione di oggetti in mano
    
    Args:
        model: MaskRCNNModel instance
        train_loader: DataLoader per i dati di addestramento
        val_loader: DataLoader per i dati di validazione
        num_epochs: Numero di epoche di addestramento
        learning_rate: Learning rate per l'ottimizzatore
        model_save_path: Percorso per salvare il miglior modello
        
    Returns:
        Dizionario con lo storico dell'addestramento
    """
    # Assicurati che le directory esistano
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    
    # Inizializza l'ottimizzatore
    params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Scheduler del learning rate
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Writer di TensorBoard con timestamp
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(TENSORBOARD_DIR, current_time)
    writer = SummaryWriter(log_dir)
    print(f"Log TensorBoard salvati in: {log_dir}")
    
    # Test del writer
    writer.add_scalar('Test/scalar', 0.0, 0)
    writer.flush()  # Forza la scrittura su disco
    
    # Storico dell'addestramento
    history = {
        'train_loss': [],
        'val_miou': [],
        'learning_rates': []
    }
    
    # Miglior mIoU di validazione
    best_miou = 0.0
    
    # Device
    device = next(model.model.parameters()).device
    print(f"Addestramento su dispositivo: {device}")
    
    # Addestra per il numero specificato di epoche
    for epoch in range(num_epochs):
        print(f"\nEpoca {epoch+1}/{num_epochs}")
        
        # ===== FASE DI TRAINING =====
        model.model.train()
        total_loss = 0.0
        loss_types = {}  # Per tracciare i tipi di loss
        
        # Progress bar per il training
        with tqdm(train_loader, desc=f"Training") as train_pbar:
            for i, (images, targets) in enumerate(train_pbar):
                # Sposta i dati sul dispositivo
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Azzera i gradienti
                optimizer.zero_grad()
                
                # Forward pass
                loss_dict = model.model(images, targets)
                
                # Calcola la loss totale
                losses = sum(loss for loss in loss_dict.values())
                
                # Accumula i valori per ogni tipo di loss
                for loss_name, loss_value in loss_dict.items():
                    if loss_name not in loss_types:
                        loss_types[loss_name] = 0.0
                    loss_types[loss_name] += loss_value.item()
                
                # Backward pass e ottimizzazione
                losses.backward()
                optimizer.step()
                
                # Aggiorna la loss totale
                total_loss += losses.item()
                
                # Aggiorna la barra di progresso
                train_pbar.set_postfix({"loss": losses.item()})
                
                # Log su TensorBoard (ogni 10 batch)
                if i % 10 == 0:
                    step = epoch * len(train_loader) + i
                    writer.add_scalar('Batch/train_loss', losses.item(), step)
                    for loss_name, loss_value in loss_dict.items():
                        writer.add_scalar(f'Batch/{loss_name}', loss_value.item(), step)
                    # Forza la scrittura
                    writer.flush()
        
        # Calcola la loss media
        avg_train_loss = total_loss / len(train_loader)
        avg_loss_types = {k: v / len(train_loader) for k, v in loss_types.items()}
        
        # Aggiorna il learning rate
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # ===== FASE DI VALIDAZIONE =====
        val_metrics = model.evaluate(val_loader)
        val_miou = val_metrics['mIoU']
        
        # Stampa le metriche
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Loss Components:")
        for k, v in avg_loss_types.items():
            print(f"  {k}: {v:.4f}")
        print(f"Validation mIoU: {val_miou:.4f}")
        print(f"Learning Rate: {current_lr}")
        
        # Aggiorna lo storico
        history['train_loss'].append(avg_train_loss)
        history['val_miou'].append(val_miou)
        history['learning_rates'].append(current_lr)
        
        # Log su TensorBoard per l'epoca
        writer.add_scalar('Epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('Epoch/val_miou', val_miou, epoch)
        writer.add_scalar('Epoch/learning_rate', current_lr, epoch)
        
        # Log dei componenti della loss
        for loss_name, avg_value in avg_loss_types.items():
            writer.add_scalar(f'Epoch/{loss_name}', avg_value, epoch)
        
        # Forza la scrittura
        writer.flush()
        
        # Salva il miglior modello
        if model_save_path and val_miou > best_miou:
            best_miou = val_miou
            model.save(model_save_path)
            print(f"Nuovo miglior modello salvato con mIoU: {best_miou:.4f}")
    
    # Assicurati di chiudere il writer di TensorBoard
    writer.close()
    print("Writer TensorBoard chiuso correttamente")
    
    return history

def main():
    # Analisi argomenti della riga di comando
    parser = argparse.ArgumentParser(description="Addestramento Mask R-CNN per segmentazione di oggetti in mano")
    parser.add_argument('--mode', type=str, choices=['train', 'continue'], default='train',
                       help='Modalità di addestramento: "train" per iniziare da zero, "continue" per continuare da un checkpoint')
    parser.add_argument('--checkpoint', type=str, default='saved_models/mask_rcnn_final.pth',
                       help='Percorso del checkpoint da cui continuare (solo per mode="continue")')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help='Numero di epoche di addestramento')
    
    args = parser.parse_args()
    
    try:
        # Prepara i dataset
        print("Inizio preparazione dataset...")
        train_dataset, val_dataset, test_dataset = prepare_datasets()
        print("Dataset preparati con successo!")
        
        # Crea i data loader
        print("Creazione data loader...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=6  
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=6  
        )
        
        print("Data loader creati con successo!")
        
        # Inizializza il modello
        print("Inizializzazione del modello Mask R-CNN...")
        
        # Scegli se usare pesi preaddestrati o no in base alla modalità
        use_pretrained = (args.mode == 'train')
        mask_rcnn = MaskRCNNModel(num_classes=NUM_CLASSES, pretrained=use_pretrained)
        
        # In modalità 'continue', carica il checkpoint
        if args.mode == 'continue':
            if os.path.exists(args.checkpoint):
                print(f"Caricamento del checkpoint da {args.checkpoint}...")
                mask_rcnn.load(args.checkpoint)
                print("Checkpoint caricato con successo!")
            else:
                print(f"ATTENZIONE: Checkpoint {args.checkpoint} non trovato. Inizializzazione con pesi DEFAULT.")
                # Reinizializza il modello con pesi preaddestrati se il checkpoint non esiste
                mask_rcnn = MaskRCNNModel(num_classes=NUM_CLASSES, pretrained=True)
        
        # Configura learning rate
        current_lr = LEARNING_RATE
        if args.mode == 'continue':
            # Riduce il learning rate se continui da un checkpoint
            current_lr = LEARNING_RATE / 10
            print(f"Learning rate ridotto a {current_lr} per il fine-tuning")
        
        # Nome del file per salvare
        if args.mode == 'train':
            model_save_path = MODEL_SAVE_PATH
            final_model_path = os.path.join('saved_models', "mask_rcnn_final.pth")
        else:
            model_save_path = os.path.join('saved_models', "mask_rcnn_continued_best.pth")
            final_model_path = os.path.join('saved_models', "mask_rcnn_final_continued.pth")
        
        # Addestra il modello
        print(f"{'Inizio' if args.mode == 'train' else 'Ripresa'} dell'addestramento per {args.epochs} epoche...")
        history = train_model(
            model=mask_rcnn,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            learning_rate=current_lr,
            model_save_path=model_save_path
        )
        
        print("Addestramento completato con successo!")
        
        # Salva esplicitamente il modello finale
        mask_rcnn.save(final_model_path)
        print(f"Modello finale salvato in: {final_model_path}")
        
        print("Processo completato con successo!")
    except Exception as e:
        print(f"Errore principale: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("Pulizia finale...")
        # Forza la liberazione di memoria
        import gc
        gc.collect()
        print("Processo train.py terminato.")

if __name__ == "__main__":    
    main()
    print("Script terminato correttamente")