import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import RESULTS_DIR, VISUALIZATIONS_DIR, MODEL_SAVE_PATH, BATCH_SIZE, NUM_CLASSES
from data.preprocessing import prepare_datasets
from data.utils import collate_fn
from models.mask_rcnn import MaskRCNNModel
from utils.visualization import visualize_prediction

def evaluate_model_on_test(model, test_loader, visualize=True, num_vis_samples=20):
    """
    Valuta il modello sul set di test con metriche dettagliate
    
    Args:
        model: Modello addestrato
        test_loader: DataLoader con i dati di test
        visualize: Se visualizzare le predizioni
        num_vis_samples: Numero di campioni da visualizzare
    
    Returns:
        Dizionario con metriche dettagliate
    """
    print("Valutazione del modello sul set di test...")
    
    # Valuta il modello
    model.model.eval()
    
    # Metriche
    all_ious = []
    empty_masks = 0
    correct_detections = 0
    total_gt_objects = 0
    total_pred_objects = 0
    
    # Per la visualizzazione
    vis_samples = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc="Valutazione")):
            # Sposta le immagini sul dispositivo
            images = [image.to(model.device) for image in images]
            
            # Ottieni le predizioni
            predictions = model.model(images)
            
            # Elabora ogni immagine
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                # Sample per visualizzazione
                if len(vis_samples) < num_vis_samples and np.random.random() < 0.1:
                    vis_samples.append((images[i], pred, target))
                
                # Sposta il target su CPU per la valutazione
                target = {k: v.cpu() for k, v in target.items()}
                
                # Ottieni le maschere predette
                pred_masks = pred['masks'].cpu()
                pred_scores = pred['scores'].cpu()
                
                # Ottieni le maschere target
                target_masks = target['masks']
                
                # Conta gli oggetti ground truth
                total_gt_objects += len(target_masks)
                
                # Salta le immagini senza maschere target
                if len(target_masks) == 0:
                    empty_masks += 1
                    continue
                
                # Converti le maschere target in formato binario (0 o 1)
                target_masks = target_masks > 0.5
                
                # Usa solo le predizioni ad alta confidenza
                high_conf_indices = pred_scores > 0.5
                high_conf_count = high_conf_indices.sum().item()
                total_pred_objects += high_conf_count
                
                if high_conf_count == 0:
                    # Nessuna predizione ad alta confidenza
                    continue
                
                pred_masks = pred_masks[high_conf_indices]
                
                # Converti le maschere predette in formato binario
                pred_masks = pred_masks > 0.5
                
                # Calcola IoU per ogni coppia di maschera predetta e target
                for pred_mask in pred_masks:
                    pred_mask = pred_mask.squeeze()
                    pred_mask_area = pred_mask.sum().item()
                    
                    # Se la maschera predetta è vuota, IoU è 0
                    if pred_mask_area == 0:
                        all_ious.append(0.0)
                        continue
                    
                    best_iou = 0.0
                    for target_mask in target_masks:
                        target_mask = target_mask.squeeze()
                        target_mask_area = target_mask.sum().item()
                        
                        # Se la maschera target è vuota, IoU è 0
                        if target_mask_area == 0:
                            continue
                        
                        # Calcola intersezione e unione
                        intersection = (pred_mask & target_mask).sum().item()
                        union = pred_mask_area + target_mask_area - intersection
                        
                        # Calcola IoU
                        iou = intersection / union if union > 0 else 0.0
                        best_iou = max(best_iou, iou)
                    
                    all_ious.append(best_iou)
                    
                    # Conta le predizioni corrette (IoU > 0.5)
                    if best_iou > 0.5:
                        correct_detections += 1
    
    # Calcola le metriche
    avg_iou = np.mean(all_ious) if all_ious else 0.0
    precision = correct_detections / total_pred_objects if total_pred_objects > 0 else 0.0
    recall = correct_detections / total_gt_objects if total_gt_objects > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Stampa i risultati
    print("\nRisultati della valutazione sul test set:")
    print(f"  mIoU: {avg_iou:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    print(f"  Totale oggetti GT: {total_gt_objects}")
    print(f"  Totale predizioni: {total_pred_objects}")
    print(f"  Predizioni corrette (IoU>0.5): {correct_detections}")
    print(f"  Immagini senza maschere: {empty_masks}")
    
    # Crea l'istogramma degli IoU
    if all_ious:
        plt.figure(figsize=(10, 6))
        plt.hist(all_ious, bins=20, alpha=0.7)
        plt.title('Distribuzione degli IoU')
        plt.xlabel('IoU')
        plt.ylabel('Frequenza')
        plt.grid(True, alpha=0.3)
        
        # Salva l'istogramma
        hist_path = os.path.join(RESULTS_DIR, "iou_histogram.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"Istogramma IoU salvato in {hist_path}")
    
    # Visualizza alcuni esempi
    if visualize and vis_samples:
        print(f"\nVisualizzazione di {len(vis_samples)} esempi...")
        vis_dir = os.path.join(VISUALIZATIONS_DIR, "test_detailed")
        os.makedirs(vis_dir, exist_ok=True)
        
        for idx, (image, pred, target) in enumerate(vis_samples):
            # Converti l'immagine in numpy
            img_tensor = image.cpu()
            img_np = img_tensor.permute(1, 2, 0).numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            
            # Ottieni maschere e scores
            pred_masks = pred['masks'].cpu()
            pred_scores = pred['scores'].cpu()
            
            # Usa solo predizioni ad alta confidenza
            high_conf_idx = pred_scores > 0.5
            pred_masks = pred_masks[high_conf_idx]
            pred_scores = pred_scores[high_conf_idx]
            
            # Converti target masks
            target_masks = target['masks'].cpu()
            
            # Crea figura con 3 subplot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Immagine originale
            axes[0].imshow(img_np)
            axes[0].set_title("Immagine originale")
            axes[0].axis('off')
            
            # Ground truth
            gt_vis = visualize_prediction(img_np, target_masks)
            axes[1].imshow(gt_vis)
            axes[1].set_title(f"Ground Truth ({len(target_masks)} oggetti)")
            axes[1].axis('off')
            
            # Predizioni
            pred_vis = visualize_prediction(img_np, pred_masks)
            axes[2].imshow(pred_vis)
            axes[2].set_title(f"Predizioni ({len(pred_masks)} oggetti)")
            axes[2].axis('off')
            
            # Aggiungi testo con scores
            if len(pred_scores) > 0:
                score_text = f"Scores: {', '.join([f'{s:.2f}' for s in pred_scores])}"
                plt.figtext(0.5, 0.01, score_text, ha="center", fontsize=10, 
                           bbox={"facecolor":"white", "alpha":0.8, "pad":5})
            
            # Salva
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"test_detailed_{idx}.png"))
            plt.close()
    
    return {
        "mIoU": avg_iou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "correct_detections": correct_detections,
        "total_gt_objects": total_gt_objects,
        "total_pred_objects": total_pred_objects,
        "empty_masks": empty_masks,
        "iou_distribution": all_ious if len(all_ious) < 1000 else []  # Limita la dimensione per il JSON
    }

def main():
    # Prepara i dataset
    _, _, test_dataset = prepare_datasets()

    # Crea il data loader per il test
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=6
    )
    
    # Inizializza il modello
    print("Inizializzazione del modello Mask R-CNN...")
    mask_rcnn = MaskRCNNModel(num_classes=NUM_CLASSES, pretrained=True)
    
    # Carica il modello addestrato
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Errore: Modello addestrato non trovato in {MODEL_SAVE_PATH}")
        return
    
    mask_rcnn.load(MODEL_SAVE_PATH)
    
    # Assicurati che la directory dei risultati esista
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Valuta il modello sul set di test
    test_metrics = evaluate_model_on_test(mask_rcnn, test_loader, visualize=True, num_vis_samples=30)
    
    print("Valutazione completata!")

if __name__ == "__main__":
    main()