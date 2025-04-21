import os
import sys
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

# Aggiungi la directory principale al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MaskRCNNModel
from data.dataset import create_data_loaders
from config import MODEL_SAVE_PATH, BATCH_SIZE, NUM_CLASSES, RESULTS_DIR

def calculate_metrics(predictions, targets, iou_threshold=0.5):
    """
    Calcola le metriche di valutazione per maschere
    
    Args:
        predictions: Lista di dizionari di predizioni
        targets: Lista di dizionari di ground truth
        iou_threshold: Soglia IoU per considerare un match
    
    Returns:
        Dizionario con metriche
    """
    # Inizializza contatori
    tp = 0  # True positive
    fp = 0  # False positive
    fn = 0  # False negative
    
    total_iou = 0.0
    total_matches = 0
    
    # Calcola TP, FP, FN
    for pred, target in zip(predictions, targets):
        pred_masks = pred['masks']  # (N, H, W)
        gt_masks = target['masks'].numpy()  # (M, H, W)
        
        # Calcola la matrice di IoU tra tutte le maschere
        if len(pred_masks) > 0 and len(gt_masks) > 0:
            ious = np.zeros((len(pred_masks), len(gt_masks)))
            for i, pred_mask in enumerate(pred_masks):
                for j, gt_mask in enumerate(gt_masks):
                    pred_mask_binary = pred_mask > 0.5
                    gt_mask_binary = gt_mask > 0.5
                    
                    # Calcola IoU: (intersection / union)
                    intersection = np.logical_and(pred_mask_binary, gt_mask_binary).sum()
                    union = np.logical_or(pred_mask_binary, gt_mask_binary).sum()
                    ious[i, j] = intersection / union if union > 0 else 0
            
            # Determina i match usando l'algoritmo greedy
            matches = []
            unmatched_preds = list(range(len(pred_masks)))
            unmatched_gts = list(range(len(gt_masks)))
            
            # Ordina tutte le coppie per IoU decrescente
            all_iou_pairs = []
            for i in range(len(pred_masks)):
                for j in range(len(gt_masks)):
                    all_iou_pairs.append((i, j, ious[i, j]))
            
            all_iou_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Trova i match
            for pred_idx, gt_idx, iou in all_iou_pairs:
                if pred_idx in unmatched_preds and gt_idx in unmatched_gts and iou >= iou_threshold:
                    matches.append((pred_idx, gt_idx, iou))
                    unmatched_preds.remove(pred_idx)
                    unmatched_gts.remove(gt_idx)
                    
                    # Aggiorna le statistiche
                    total_iou += iou
                    total_matches += 1
            
            # Aggiorna i contatori
            tp += len(matches)
            fp += len(unmatched_preds)
            fn += len(unmatched_gts)
        else:
            # Se non ci sono predizioni ma ci sono ground truth
            if len(gt_masks) > 0:
                fn += len(gt_masks)
            # Se ci sono predizioni ma non ci sono ground truth
            if len(pred_masks) > 0:
                fp += len(pred_masks)
    
    # Calcola le metriche
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    mean_iou = total_iou / total_matches if total_matches > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_iou": mean_iou,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }

def evaluate_model(args):
    """
    Funzione principale per la valutazione del modello
    
    Args:
        args: Argomenti da riga di comando
    """
    print("=" * 50)
    print("VALUTAZIONE MASK R-CNN PER HOT3D DATASET")
    print("=" * 50)
    
    # Verifica disponibilità della GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Inizializza il modello
    print("\nInizializzazione modello...")
    model = MaskRCNNModel(
        num_classes=args.num_classes,
        pretrained=False,  # Non è necessario pretrained per l'eval
        backbone_name=args.backbone
    )
    
    # Carica il modello
    if not model.load(args.model):
        print(f"ERRORE: Impossibile caricare il modello da {args.model}")
        return
    
    # Crea il dataloader per il test
    print("\nCreazione dataloader...")
    try:
        if args.dataset_type == 'train':
            train_loader, _, _ = create_data_loaders(batch_size=1)  # batch_size=1 per l'eval
            eval_loader = train_loader
        elif args.dataset_type == 'val':
            _, eval_loader, _ = create_data_loaders(batch_size=1)
        else:  # test
            _, _, eval_loader = create_data_loaders(batch_size=1)
        
        print(f"Dataloader creato con {len(eval_loader)} batch")
    except Exception as e:
        print(f"ERRORE nella creazione del dataloader: {str(e)}")
        return
    
    # Avvia valutazione
    print("\nAvvio valutazione...")
    start_time = datetime.now()
    
    # Imposta il modello in modalità valutazione
    model.model.eval()
    
    all_preds = []
    all_targets = []
    
    # Disabilita il calcolo dei gradienti
    with torch.no_grad():
        for images, targets in tqdm(eval_loader, desc="Valutazione"):
            # Sposta i dati sul device
            images = [image.to(model.device) for image in images]
            
            # Per calcolare le metriche, mantieni i target sulla CPU
            target_cpu = targets
            
            # Forward pass per le predizioni
            outputs = model.model(images)
            
            # Converti le predizioni in un formato utilizzabile
            for output, target in zip(outputs, target_cpu):
                # Estrai le predizioni
                pred = {
                    'boxes': output['boxes'].cpu(),
                    'scores': output['scores'].cpu(),
                    'labels': output['labels'].cpu(),
                    'masks': output['masks'].cpu()  # Mantieni la dimensione del canale
                }
                
                # Applica la soglia di confidenza
                keep = pred['scores'] > args.threshold
                pred['boxes'] = pred['boxes'][keep]
                pred['scores'] = pred['scores'][keep]
                pred['labels'] = pred['labels'][keep]
                pred['masks'] = pred['masks'][keep]
                
                # Aggiungi alle liste
                all_preds.append(pred)
                all_targets.append(target)
    
    # Calcola mIoU utilizzando la funzione della classe model
    miou = model.calculate_miou(all_preds, all_targets, mask_binarize_threshold=0.5)
    
    # Calcola anche le altre metriche
    metrics = calculate_metrics(all_preds, all_targets, iou_threshold=args.iou_threshold)
    
    # Aggiungi mIoU alle metriche
    metrics['miou'] = miou
    
    # Stampa le metriche
    print("\nMetriche di valutazione:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"mIoU: {metrics['miou']:.4f}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    
    # Salva i risultati
    results_file = os.path.join(
        RESULTS_DIR, 
        f"eval_{args.dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Aggiungi informazioni aggiuntive
    results = {
        "model_path": args.model,
        "dataset_type": args.dataset_type,
        "threshold": args.threshold,
        "iou_threshold": args.iou_threshold,
        "metrics": metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Stampa tempo di valutazione
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nValutazione completata in {duration}")
    print(f"Risultati salvati in: {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Valutazione Mask R-CNN per HOT3D dataset')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Percorso al modello da valutare')
    parser.add_argument('--dataset_type', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Tipo di dataset da utilizzare per la valutazione')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Soglia di confidenza per le predizioni')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='Soglia IoU per considerare un match')
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES,
                        help='Numero di classi (incluso background)')
    parser.add_argument('--backbone', type=str, default="resnext101_32x8d",
                        choices=["resnext101_32x8d", "resnet50"],
                        help='Tipo di backbone utilizzato nel modello')
    
    args = parser.parse_args()
    evaluate_model(args)

if __name__ == "__main__":
    main()


