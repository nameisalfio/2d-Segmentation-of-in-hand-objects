import os
import sys
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import run_train
from evaluate import run_evaluation
from inference import run_inference
from config import * 

def main():
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for in-hand object segmentation on the Visor dataset.', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )
    
    # Main commands (train, evaluate, inference)
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    subparsers.required = True # Ensures a command must be provided
    
    # --- Training command ---
    train_parser = subparsers.add_parser('train', help='Train the Mask R-CNN model')
    train_parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training.')
    train_parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Initial learning rate for the optimizer.')
    train_parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Total number of training epochs.')
    train_parser.add_argument('--num_classes', type=int, default=NUM_CLASSES,
                        help='Number of classes including the background (e.g., N_actual_classes + 1).')
    train_parser.add_argument('--backbone', type=str, default="resnext101_32x8d",
                        choices=["resnext101_32x8d", "resnet50"],
                        help='Backbone architecture to use for the Mask R-CNN model.')
    train_parser.add_argument('--no_pretrained', action='store_true',
                        help='If set, do not use pretrained weights for the backbone (e.g., from ImageNet).')
    train_parser.add_argument('--output', type=str, default=MODEL_SAVE_PATH,
                        help='Path where the trained model checkpoints will be saved.')
    train_parser.add_argument('--resume', type=str, default=None,
                        help='Path to a model checkpoint to resume training from.')
    train_parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Maximum norm for gradient clipping (set to 0 to disable).')
    train_parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'],
                        help='Optimizer to use during training.')
    
    # --- Evaluation command ---
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained Mask R-CNN model')
    eval_parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file (.pth) to evaluate.')
    eval_parser.add_argument('--dataset_type', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to use for evaluation (train, val, or test).')
    eval_parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence score threshold for considering a prediction.')
    eval_parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for considering a predicted mask a true positive match.')
    eval_parser.add_argument('--num_classes', type=int, default=NUM_CLASSES,
                        help='Number of classes (including background) the model was trained with.')
    eval_parser.add_argument('--backbone', type=str, default="resnext101_32x8d",
                        choices=["resnext101_32x8d", "resnet50"],
                        help='Backbone architecture of the model being evaluated (must match the loaded model).')
    
    # --- Inference command ---
    inference_parser = subparsers.add_parser('inference', help='Run inference with a trained model on images')
    inference_parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file (.pth) to use for inference.')
    inference_parser.add_argument('--input', type=str, required=True,
                        help='Path to an input image or a directory containing images.')
    inference_parser.add_argument('--output', type=str, default='inference_results',
                        help='Directory where inference results (annotated images) will be saved.')
    inference_parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence score threshold for displaying predictions.')
    inference_parser.add_argument('--num_classes', type=int, default=NUM_CLASSES,
                        help='Number of classes (including background) the model was trained with.')
    inference_parser.add_argument('--class_names', type=str, default=None,
                        help='Comma-separated class names for visualization (e.g., "background,obj1,obj2"). Order should match model output.')
    inference_parser.add_argument('--show', action='store_true',
                        help='If set, display processed images (only applicable if the input is a single image).')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Timestamp for logging execution start
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Executing command: {args.command}")
    
    if args.command == 'train':
        print("Starting model training...")
        run_train(args)
    
    elif args.command == 'evaluate':
        print("Starting model evaluation...")
        run_evaluation(args)
    
    elif args.command == 'inference':
        print("Starting model inference...")
        run_inference(args) 
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Command '{args.command}' completed.")

if __name__ == "__main__":
    main()