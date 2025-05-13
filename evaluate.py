import os
import sys
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

# Add the main directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MaskRCNNModel
from data.dataset import create_data_loaders
from config import MODEL_SAVE_PATH, BATCH_SIZE, NUM_CLASSES, RESULTS_DIR

def calculate_metrics(predictions, targets, iou_threshold=0.5):
    """
    Calculates evaluation metrics for masks.
    
    Args:
        predictions: List of prediction dictionaries.
        targets: List of ground truth dictionaries.
        iou_threshold: IoU threshold to consider a match.
    
    Returns:
        Dictionary with metrics.
    """
    # Initialize counters
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    total_iou = 0.0
    total_matches = 0
    
    # Calculate TP, FP, FN
    for pred, target in zip(predictions, targets):
        pred_masks = pred['masks']  # (N, H, W)
        gt_masks = target['masks'].numpy()  # (M, H, W)
        
        # Calculate the IoU matrix between all masks
        if len(pred_masks) > 0 and len(gt_masks) > 0:
            ious = np.zeros((len(pred_masks), len(gt_masks)))
            for i, pred_mask in enumerate(pred_masks):
                for j, gt_mask in enumerate(gt_masks):
                    pred_mask_binary = pred_mask > 0.5
                    gt_mask_binary = gt_mask > 0.5
                    
                    # Calculate IoU: (intersection / union)
                    intersection = np.logical_and(pred_mask_binary, gt_mask_binary).sum()
                    union = np.logical_or(pred_mask_binary, gt_mask_binary).sum()
                    ious[i, j] = intersection / union if union > 0 else 0
            
            # Determine matches using a greedy algorithm
            matches = []
            unmatched_preds = list(range(len(pred_masks)))
            unmatched_gts = list(range(len(gt_masks)))
            
            # Sort all pairs by IoU in descending order
            all_iou_pairs = []
            for i in range(len(pred_masks)):
                for j in range(len(gt_masks)):
                    all_iou_pairs.append((i, j, ious[i, j]))
            
            all_iou_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Find matches
            for pred_idx, gt_idx, iou in all_iou_pairs:
                if pred_idx in unmatched_preds and gt_idx in unmatched_gts and iou >= iou_threshold:
                    matches.append((pred_idx, gt_idx, iou))
                    unmatched_preds.remove(pred_idx)
                    unmatched_gts.remove(gt_idx)
                    
                    total_iou += iou
                    total_matches += 1
            
            tp += len(matches)
            fp += len(unmatched_preds)
            fn += len(unmatched_gts)
        else:
            # If there are no predictions but there are ground truths
            if len(gt_masks) > 0:
                fn += len(gt_masks)
            # If there are predictions but no ground truths
            if len(pred_masks) > 0:
                fp += len(pred_masks)
    
    # Calculate metrics
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

def run_evaluation(args):
    """
    Main function for model evaluation.
    
    Args:
        args: Command-line arguments.
    """
    print("=" * 50)
    print("MASK R-CNN EVALUATION FOR HOT3D DATASET") # Or the relevant dataset name
    print("=" * 50)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize the model
    print("\nInitializing model...")
    model = MaskRCNNModel(
        num_classes=args.num_classes,
        pretrained=False,  # Pretrained weights are not necessary for evaluation
        backbone_name=args.backbone
    )
    
    # Load the model
    if not model.load(args.model):
        print(f"ERROR: Could not load model from {args.model}")
        return
    
    # Create the dataloader for testing
    print("\nCreating dataloader...")
    try:
        if args.dataset_type == 'train':
            train_loader, _, _ = create_data_loaders(batch_size=1)  # batch_size=1 for evaluation
            eval_loader = train_loader
        elif args.dataset_type == 'val':
            _, eval_loader, _ = create_data_loaders(batch_size=1)
        else:  # test
            _, _, eval_loader = create_data_loaders(batch_size=1)
        
        print(f"Dataloader created with {len(eval_loader)} batches")
    except Exception as e:
        print(f"ERROR creating dataloader: {str(e)}")
        return
    
    # Start evaluation
    print("\nStarting evaluation...")
    start_time = datetime.now()
    
    model.model.eval() # Set model to evaluation mode
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad(): # Disable gradient calculations
        for images, targets in tqdm(eval_loader, desc="Evaluating"):
            # Move data to the device
            images = [image.to(model.device) for image in images]
            
            # Keep targets on CPU for metric calculation
            target_cpu = targets
            
            outputs = model.model(images) # Forward pass for predictions
            
            # Convert predictions to a usable format
            for output, target in zip(outputs, target_cpu):
                pred = {
                    'boxes': output['boxes'].cpu(),
                    'scores': output['scores'].cpu(),
                    'labels': output['labels'].cpu(),
                    'masks': output['masks'].cpu()  # Keep the channel dimension
                }
                
                # Apply confidence threshold
                keep = pred['scores'] > args.threshold
                pred['boxes'] = pred['boxes'][keep]
                pred['scores'] = pred['scores'][keep]
                pred['labels'] = pred['labels'][keep]
                pred['masks'] = pred['masks'][keep]
                
                all_preds.append(pred)
                all_targets.append(target)
    
    # Calculate mIoU using the model class's function
    miou = model.calculate_miou(all_preds, all_targets, mask_binarize_threshold=0.5)
    
    # Calculate other metrics
    metrics = calculate_metrics(all_preds, all_targets, iou_threshold=args.iou_threshold)
    
    metrics['miou'] = miou # Add mIoU to the metrics
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Mean IoU (from custom calculation): {metrics['mean_iou']:.4f}") # Clarified which mean_iou this is
    print(f"mIoU (from model method): {metrics['miou']:.4f}") # Clarified which mIoU this is
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    
    # Save results
    results_file = os.path.join(
        RESULTS_DIR, 
        f"eval_{args.dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Add additional information to results
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
    
    # Print evaluation time
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nEvaluation completed in {duration}")
    print(f"Results saved to: {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Mask R-CNN Evaluation for HOT3D dataset') # Or relevant dataset
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the model to evaluate')
    parser.add_argument('--dataset_type', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Type of dataset to use for evaluation')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold to consider a match')
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES,
                        help='Number of classes (including background)')
    parser.add_argument('--backbone', type=str, default="resnext101_32x8d",
                        choices=["resnext101_32x8d", "resnet50"],
                        help='Type of backbone used in the model')
    
    args = parser.parse_args()
    run_evaluation(args)

if __name__ == "__main__":
    main()