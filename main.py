#!/usr/bin/env python3
"""
Script principale per l'addestramento, la valutazione e l'inferenza 
del modello Mask R-CNN sul dataset HOT3D per la segmentazione di oggetti in mano.
"""

import os
import sys
import argparse
from datetime import datetime

# Aggiungi la directory principale al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa i moduli
from train import train_model
from evaluate import evaluate_model
from inference import run_inference
from data.dataset import build_dataset_files
from config import MODEL_SAVE_PATH, NUM_CLASSES, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE

def main():
    parser = argparse.ArgumentParser(
        description='Mask R-CNN per la segmentazione di oggetti in mano nel dataset HOT3D',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Comandi principali
    subparsers = parser.add_subparsers(dest='command', help='Comando da eseguire')
    subparsers.required = True
    
    # Comando per il preprocessing dei dati
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocessing del dataset')
    preprocess_parser.add_argument('--dataset_type', choices=['train', 'test', 'both'], default='both',
                        help='Tipo di dataset da processare')
    preprocess_parser.add_argument('--max_clips', default='all', 
                        help='Numero massimo di clip da processare (o "all" per tutti)')
    preprocess_parser.add_argument('--max_frames', type=int, default=None,
                        help='Numero massimo di frame da selezionare randomicamente per ogni clip')
    preprocess_parser.add_argument('--debug', action='store_true', help='Salva immagini di debug')
    preprocess_parser.add_argument('--cameras', nargs='+', default=None, help='ID telecamere da processare')
    
    # Comando per l'addestramento
    train_parser = subparsers.add_parser('train', help='Addestramento del modello')
    train_parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Dimensione del batch')
    train_parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate iniziale')
    train_parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Numero di epoche di addestramento')
    train_parser.add_argument('--num_classes', type=int, default=NUM_CLASSES,
                        help='Numero di classi (incluso background)')
    train_parser.add_argument('--backbone', type=str, default="resnext101_32x8d",
                        choices=["resnext101_32x8d", "resnet50"],
                        help='Tipo di backbone da utilizzare')
    train_parser.add_argument('--no_pretrained', action='store_true',
                        help='Non utilizzare pesi preaddestrati')
    train_parser.add_argument('--output', type=str, default=MODEL_SAVE_PATH,
                        help='Percorso per il salvataggio del modello')
    train_parser.add_argument('--resume', type=str, default=None,
                        help='Riprendi addestramento da checkpoint')
    train_parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Valore per il clipping dei gradienti (0 per disabilitare)')
    train_parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'],
                        help='Ottimizzatore da utilizzare')
    
    # Comando per la valutazione
    eval_parser = subparsers.add_parser('evaluate', help='Valutazione del modello')
    eval_parser.add_argument('--model', type=str, required=True,
                        help='Percorso al modello da valutare')
    eval_parser.add_argument('--dataset_type', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Tipo di dataset da utilizzare per la valutazione')
    eval_parser.add_argument('--threshold', type=float, default=0.5,
                        help='Soglia di confidenza per le predizioni')
    eval_parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='Soglia IoU per considerare un match')
    eval_parser.add_argument('--num_classes', type=int, default=NUM_CLASSES,
                        help='Numero di classi (incluso background)')
    eval_parser.add_argument('--backbone', type=str, default="resnext101_32x8d",
                        choices=["resnext101_32x8d", "resnet50"],
                        help='Tipo di backbone utilizzato nel modello')
    
    # Comando per l'inferenza
    inference_parser = subparsers.add_parser('inference', help='Inferenza del modello')
    inference_parser.add_argument('--model', type=str, required=True,
                        help='Percorso al modello da utilizzare')
    inference_parser.add_argument('--input', type=str, required=True,
                        help='Percorso all\'immagine o directory di immagini')
    inference_parser.add_argument('--output', type=str, default='inference_results',
                        help='Directory di output per i risultati')
    inference_parser.add_argument('--threshold', type=float, default=0.5,
                        help='Soglia di confidenza per le predizioni')
    inference_parser.add_argument('--num_classes', type=int, default=NUM_CLASSES,
                        help='Numero di classi (incluso background)')
    parser.add_argument('--class_names', type=str, default=None,
                        help='Nomi delle classi separati da virgola')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Soglia di confidenza per le predizioni')
    parser.add_argument('--show', action='store_true',
                        help='Mostra le immagini elaborate (solo per singola immagine)')
    
    # Comando per la visualizzazione
    visualize_parser = subparsers.add_parser('visualize', help='Visualizzazione campioni dataset')
    visualize_parser.add_argument('--dataset', choices=['train', 'val', 'test'], default='train',
                        help='Dataset da visualizzare')
    visualize_parser.add_argument('--num_samples', type=int, default=10,
                        help='Numero di campioni da visualizzare')
    visualize_parser.add_argument('--output_dir', default='sample_visualizations',
                        help='Directory dove salvare le visualizzazioni')
    visualize_parser.add_argument('--seed', type=int, default=42,
                        help='Seed per la selezione random dei campioni')
    
    # Parsing degli argomenti
    args = parser.parse_args()
    
    # Timestamp per log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Esecuzione comando: {args.command}")
    
    # Esegui il comando appropriato
    if args.command == 'preprocess':
        print("Avvio preprocessing dataset...")
        process_train = args.dataset_type in ['train', 'both']
        process_test = args.dataset_type in ['test', 'both']
        build_dataset_files(process_train, process_test, args.max_clips, args.debug, args.max_frames)
    
    elif args.command == 'train':
        print("Avvio addestramento modello...")
        train_model(args)
    
    elif args.command == 'evaluate':
        print("Avvio valutazione modello...")
        evaluate_model(args)
    
    elif args.command == 'inference':
        print("Avvio inferenza modello...")
        run_inference(args)
    
    elif args.command == 'visualize':
        print("Avvio visualizzazione dataset...")
        # Importa qui per evitare dipendenze circolari
        from visualize_samples import visualize_samples
        visualize_samples(
            dataset_path=f"dataset_cache/{args.dataset}_dataset.npy",
            output_dir=f"{args.output_dir}/{args.dataset}",
            num_samples=args.num_samples,
            random_seed=args.seed
        )
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Comando {args.command} completato.")

if __name__ == "__main__":
    main()