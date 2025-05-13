#!/usr/bin/env python3
"""
Script principale per l'addestramento, la valutazione e l'inferenza 
del modello Mask R-CNN sul dataset Visor per la segmentazione di oggetti in mano.
"""

import os
import sys
import argparse
from datetime import datetime

# Aggiungi la directory principale al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa i moduli
from train import run_train
from evaluate import run_evaluation
from inference import run_inference
from config import *

def main():
    parser = argparse.ArgumentParser(
        description='Mask R-CNN per la segmentazione di oggetti in mano nel dataset Visor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Comandi principali
    subparsers = parser.add_subparsers(dest='command', help='Comando da eseguire')
    subparsers.required = True
    
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
    
    # Parsing degli argomenti
    args = parser.parse_args()
    
    # Timestamp per log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Esecuzione comando: {args.command}")
    
    if args.command == 'train':
        print("Avvio addestramento modello...")
        run_train(args)
    
    elif args.command == 'evaluate':
        print("Avvio valutazione modello...")
        run_evaluation(args)
    
    elif args.command == 'inference':
        print("Avvio inferenza modello...")
        run_inference(args)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Comando {args.command} completato.")

if __name__ == "__main__":
    main()