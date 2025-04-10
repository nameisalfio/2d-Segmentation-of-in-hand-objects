import torch
import numpy as np
from tqdm import tqdm

def calculate_iou(pred_mask, gt_mask):
    """
    Calcola l'Intersection over Union (IoU) tra una maschera predetta e una ground truth
    
    Args:
        pred_mask: Maschera predetta (tensore binario)
        gt_mask: Maschera ground truth (tensore binario)
        
    Returns:
        IoU calcolato
    """
    # Assicurati che entrambe le maschere siano binarie
    if pred_mask.dtype != torch.bool:
        pred_mask = pred_mask > 0.5
    
    if gt_mask.dtype != torch.bool:
        gt_mask = gt_mask > 0.5
    
    # Calcola l'area della maschera predetta
    pred_area = pred_mask.sum().item()
    
    # Calcola l'area della maschera ground truth
    gt_area = gt_mask.sum().item()
    
    # Se entrambe le maschere sono vuote, IoU è 1
    if pred_area == 0 and gt_area == 0:
        return 1.0
    
    # Se una delle maschere è vuota, IoU è 0
    if pred_area == 0 or gt_area == 0:
        return 0.0
    
    # Calcola intersezione
    intersection = (pred_mask & gt_mask).sum().item()
    
    # Calcola unione
    union = pred_area + gt_area - intersection
    
    # Calcola IoU
    iou = intersection / union
    
    return iou

def evaluate_model(model, data_loader, iou_threshold=0.5):
    """
    Valuta il modello su un insieme di dati
    
    Args:
        model: Modello da valutare
        data_loader: DataLoader con i dati di validazione/test
        iou_threshold: Soglia IoU per considerare una predizione corretta
        
    Returns:
        Dizionario con mIoU
    """
    # Ottieni le metriche dal modello
    metrics = model.evaluate(data_loader, iou_threshold)
    
    # Stampa le metriche
    print(f"Metriche di valutazione:")
    print(f"  mIoU: {metrics['mIoU']:.4f}")
    
    return metrics