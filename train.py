import os
import sys
import argparse
import torch
from datetime import datetime

# Aggiungi la directory principale al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MaskRCNNModel
from data.dataset import create_data_loaders
from config import MODEL_SAVE_PATH, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, NUM_CLASSES, DATASET_CACHE_DIR, MODELS_DIR

def run_train(args):
    """
    Funzione principale per l'addestramento del modello
    
    Args:
        args: Argomenti da riga di comando
    """
    print("=" * 50)
    print("ADDESTRAMENTO MASK R-CNN")
    print("=" * 50)
    
    # Verifica disponibilità della GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    if device.type == 'cuda':
        gpu_properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_properties.name}")
        
        # Calcola memoria disponibile in GB
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        total_memory = gpu_properties.total_memory / (1024**3)
        available_memory = total_memory - memory_allocated
        print(f"Memoria GPU disponibile: {available_memory:.2f} GB")
    
    # Crea i dataloader
    print("\nCreazione dataloader...")
    try:
        train_loader, val_loader, _ = create_data_loaders(batch_size=args.batch_size)
        print(f"Dataloader creati: {len(train_loader)} batch di training, {len(val_loader)} batch di validation")
    except Exception as e:
        print(f"ERRORE nella creazione dei dataloader: {str(e)}")
        return
    
    # Inizializzazione modello
    print("\nInizializzazione modello...")
    
    # Gestisci il caso in cui gli attributi non esistano
    clip_grad_value = 1.0  # Valore predefinito
    optimizer_type = 'sgd'  # Valore predefinito
    
    # Verifica quali attributi sono disponibili
    if hasattr(args, 'clip_grad'):
        clip_grad_value = args.clip_grad
    elif hasattr(args, 'clip_grad_norm'):
        clip_grad_value = args.clip_grad_norm
        
    if hasattr(args, 'optimizer'):
        optimizer_type = args.optimizer
    
    # Modifica alla creazione del modello, passando parametri aggiuntivi
    model = MaskRCNNModel(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        backbone_name=args.backbone,
        clip_grad_norm=clip_grad_value,
        optimizer_type=optimizer_type
    )
    
    # Carica checkpoint se specificato
    if args.resume and os.path.exists(args.resume):
        print(f"Ripresa addestramento da checkpoint: {args.resume}")
        model.load(args.resume)
    
    # Crea directory del modello se non esiste
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Configurazione addestramento
    print("\nConfigurazione addestramento:")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Learning rate: {args.lr}")
    print(f"- Epoche: {args.epochs}")
    print(f"- Backbone: {args.backbone}")
    print(f"- Pretrained: {not args.no_pretrained}")
    print(f"- Output: {args.output}")
    print(f"- Clip gradiente: {clip_grad_value}")
    print(f"- Ottimizzatore: {optimizer_type}")
    
    # Avvia addestramento
    print("\nAvvio addestramento...")
    start_time = datetime.now()
    
    try:
        history = model.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            model_save_path=args.output
        )
        
        # Stampa tempo di addestramento
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\nAddestramento completato in {duration}")
        print(f"Modello salvato in: {args.output}")
        
        # Stampa metriche finali
        if history["val_miou"]:
            best_map = max(history["val_miou"])
            best_epoch = history["val_miou"].index(best_map) + 1
            print(f"Miglior mIoU: {best_map:.4f} (Epoca {best_epoch})")
    
    except KeyboardInterrupt:
        print("\nAddestramento interrotto dall'utente")
        print("Salvataggio modello parziale...")
        model.save(args.output.replace(".pth", "_partial.pth"))
    
    except Exception as e:
        print(f"\nERRORE durante l'addestramento: {str(e)}")
        # Salva modello di emergenza
        emergency_path = os.path.join(MODELS_DIR, "emergency_save.pth")
        model.save(emergency_path)
        print(f"Modello di emergenza salvato in: {emergency_path}")

def main():
    parser = argparse.ArgumentParser(description='Addestramento Mask R-CNN')
    
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Dimensione del batch')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate iniziale')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Numero di epoche di addestramento')
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES,
                        help='Numero di classi (incluso background)')
    parser.add_argument('--backbone', type=str, default="resnext101_32x8d",
                        choices=["resnext101_32x8d", "resnet50"],
                        help='Tipo di backbone da utilizzare')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Non utilizzare pesi preaddestrati')
    parser.add_argument('--output', type=str, default=MODEL_SAVE_PATH,
                        help='Percorso per il salvataggio del modello')
    parser.add_argument('--resume', type=str, default=None,
                        help='Riprendi addestramento da checkpoint')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Valore per il clipping dei gradienti (0 per disabilitare)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'],
                        help='Ottimizzatore da utilizzare')
    
    args = parser.parse_args()
    run_train(args)

if __name__ == "__main__":
    main()