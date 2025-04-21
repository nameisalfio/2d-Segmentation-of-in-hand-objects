#!/usr/bin/env python3
"""
Script per creare versioni ridotte dei file dataset NPY.
Questo script carica i file dataset originali e crea nuove versioni con un 
numero ridotto di campioni, utile per test e debugging.
"""

import os
import sys
import numpy as np
import argparse
import random
from tqdm import tqdm

# Aggiungi la directory principale al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def reduce_dataset(input_file, output_file, num_samples, random_selection=False):
    """
    Crea una versione ridotta di un file dataset NPY.
    
    Args:
        input_file: Path al file NPY originale
        output_file: Path dove salvare il file NPY ridotto
        num_samples: Numero di campioni da includere nel file ridotto
        random_selection: Se True, seleziona i campioni casualmente, 
                          altrimenti prende i primi 'num_samples'
    """
    print(f"Caricamento dataset originale da {input_file}...")
    original_dataset = np.load(input_file, allow_pickle=True)
    
    total_samples = len(original_dataset)
    print(f"Dataset originale contiene {total_samples} campioni")
    
    # Verifica che num_samples sia valido
    if num_samples >= total_samples:
        print(f"ATTENZIONE: Numero di campioni richiesto ({num_samples}) è maggiore o uguale al totale ({total_samples})")
        print("Verrà creata una copia completa del dataset originale")
        reduced_dataset = original_dataset
    else:
        # Seleziona campioni
        if random_selection:
            print(f"Selezione casuale di {num_samples} campioni...")
            indices = random.sample(range(total_samples), num_samples)
            reduced_dataset = original_dataset[indices]
        else:
            print(f"Selezione dei primi {num_samples} campioni...")
            reduced_dataset = original_dataset[:num_samples]
    
    # Crea directory di output se necessario
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Salva il dataset ridotto
    print(f"Salvataggio dataset ridotto in {output_file}...")
    np.save(output_file, reduced_dataset)
    
    print(f"Dataset ridotto salvato con successo: {len(reduced_dataset)} campioni")
    return len(reduced_dataset)

def check_valid_samples(dataset_file):
    """
    Verifica quanti campioni validi ci sono nel dataset.
    Un campione è considerato valido se contiene un'immagine esistente.
    
    Args:
        dataset_file: Path al file NPY del dataset
        
    Returns:
        tuple: (num_valid, num_total, list_of_invalid_indices)
    """
    print(f"Controllo validità campioni in {dataset_file}...")
    dataset = np.load(dataset_file, allow_pickle=True)
    
    invalid_indices = []
    for i, sample in enumerate(tqdm(dataset)):
        if "image_path" not in sample or not os.path.exists(sample["image_path"]):
            invalid_indices.append(i)
    
    num_valid = len(dataset) - len(invalid_indices)
    print(f"Campioni totali: {len(dataset)}")
    print(f"Campioni validi: {num_valid}")
    print(f"Campioni non validi: {len(invalid_indices)}")
    
    return num_valid, len(dataset), invalid_indices

def main():
    parser = argparse.ArgumentParser(description='Crea versioni ridotte dei file dataset NPY')
    parser.add_argument('--input', type=str, help='File di input. Se non specificato, usa i percorsi predefiniti')
    parser.add_argument('--output', type=str, help='File di output. Se non specificato, usa i percorsi predefiniti')
    parser.add_argument('--samples', type=int, default=100, help='Numero di campioni da includere nel dataset ridotto')
    parser.add_argument('--random', action='store_true', help='Seleziona campioni casualmente invece dei primi N')
    parser.add_argument('--check-only', action='store_true', help='Controlla solo il numero di campioni validi senza creare il dataset ridotto')
    parser.add_argument('--all', action='store_true', help='Crea versioni ridotte di tutti i dataset (train, val, test)')
    
    args = parser.parse_args()
    
    # Se --check-only è specificato, controlla solo la validità
    if args.check_only:
        if args.input:
            check_valid_samples(args.input)
        elif args.all:
            for dataset_path in [TRAIN_CACHE_PATH, VAL_CACHE_PATH, TEST_CACHE_PATH]:
                if os.path.exists(dataset_path):
                    check_valid_samples(dataset_path)
        else:
            check_valid_samples(TRAIN_CACHE_PATH)
        return
    
    # Determina quali dataset ridurre
    if args.all:
        datasets_to_reduce = []
        if os.path.exists(TRAIN_CACHE_PATH):
            datasets_to_reduce.append((TRAIN_CACHE_PATH, TRAIN_CACHE_PATH.replace('.npy', f'_reduced_{args.samples}.npy')))
        if os.path.exists(VAL_CACHE_PATH):
            datasets_to_reduce.append((VAL_CACHE_PATH, VAL_CACHE_PATH.replace('.npy', f'_reduced_{args.samples}.npy')))
        if os.path.exists(TEST_CACHE_PATH):
            datasets_to_reduce.append((TEST_CACHE_PATH, TEST_CACHE_PATH.replace('.npy', f'_reduced_{args.samples}.npy')))
    elif args.input:
        input_file = args.input
        if args.output:
            output_file = args.output
        else:
            output_file = input_file.replace('.npy', f'_reduced_{args.samples}.npy')
        datasets_to_reduce = [(input_file, output_file)]
    else:
        # Default: riduci solo il dataset di training
        input_file = TRAIN_CACHE_PATH
        output_file = TRAIN_CACHE_PATH.replace('.npy', f'_reduced_{args.samples}.npy')
        datasets_to_reduce = [(input_file, output_file)]
    
    # Riduci ciascun dataset
    total_processed = 0
    for input_file, output_file in datasets_to_reduce:
        if os.path.exists(input_file):
            print(f"\nProcessing: {input_file}")
            samples_processed = reduce_dataset(input_file, output_file, args.samples, args.random)
            total_processed += samples_processed
        else:
            print(f"ERRORE: File di input non trovato: {input_file}")
    
    if total_processed > 0:
        print(f"\nCreazione completata: {total_processed} campioni totali salvati in versioni ridotte")
        print("\nPer utilizzare i dataset ridotti:")
        print("1. Modifica i percorsi in config.py per puntare ai file ridotti, oppure")
        print("2. Rinomina i file ridotti con i nomi originali\n")

if __name__ == "__main__":
    main()