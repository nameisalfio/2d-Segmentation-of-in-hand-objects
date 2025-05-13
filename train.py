import os
import sys
import argparse
import torch
from datetime import datetime

# Add the parent directory to sys.path to allow imports from sibling modules/packages
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MaskRCNNModel
from data.dataset import create_data_loaders
from config import MODEL_SAVE_PATH, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, NUM_CLASSES, DATASET_CACHE_DIR, MODELS_DIR

def run_train(args):
    """
    Main function for model training.
    
    Args:
        args: Command-line arguments.
    """
    print("=" * 50)
    print("MASK R-CNN TRAINING")
    print("=" * 50)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        gpu_properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_properties.name}")
        
        # Calculate available GPU memory in GB
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        total_memory = gpu_properties.total_memory / (1024**3)
        available_memory = total_memory - memory_allocated
        print(f"Available GPU Memory: {available_memory:.2f} GB")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    try:
        train_loader, val_loader, _ = create_data_loaders(batch_size=args.batch_size)
        print(f"Dataloaders created: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    except Exception as e:
        print(f"ERROR creating dataloaders: {str(e)}")
        return
    
    # Initialize model
    print("\nInitializing model...")
    
    clip_grad_value = 1.0  
    optimizer_type = 'sgd'  
    
    if hasattr(args, 'clip_grad'):
        clip_grad_value = args.clip_grad
    elif hasattr(args, 'clip_grad_norm'): 
        clip_grad_value = args.clip_grad_norm
        
    if hasattr(args, 'optimizer'):
        optimizer_type = args.optimizer
    
    # Instantiate the model, passing additional parameters
    model = MaskRCNNModel(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        backbone_name=args.backbone,
        clip_grad_norm=clip_grad_value, 
        optimizer_type=optimizer_type   
    )
    
    # Load checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        model.load(args.resume) 
    
    # Create model output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Training configuration
    print("\nTraining Configuration:")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Learning rate: {args.lr}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Backbone: {args.backbone}")
    print(f"- Pretrained backbone: {not args.no_pretrained}")
    print(f"- Output model path: {args.output}")
    print(f"- Gradient clipping norm: {clip_grad_value}")
    print(f"- Optimizer: {optimizer_type}")
    
    # Start training
    print("\nStarting training...")
    start_time = datetime.now()
    
    try:
        history = model.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            model_save_path=args.output
        )
        
        # Print training time
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\nTraining completed in {duration}")
        print(f"Model saved to: {args.output}")
        
        if history and "val_miou" in history and history["val_miou"]: 
            best_miou = max(history["val_miou"])
            best_epoch = history["val_miou"].index(best_miou) + 1
            print(f"Best validation mIoU: {best_miou:.4f} (Epoch {best_epoch})")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("Saving partial model...")
        partial_model_path = args.output.replace(".pth", "_partial.pth")
        model.save(partial_model_path)
        print(f"Partial model saved to: {partial_model_path}")
    
    except Exception as e:
        print(f"\nERROR during training: {str(e)}")
        # Save emergency model
        emergency_path = os.path.join(MODELS_DIR, f"emergency_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        model.save(emergency_path)
        print(f"Emergency model saved to: {emergency_path}")

def main():
    parser = argparse.ArgumentParser(description='Mask R-CNN Training Script')
    
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs.')
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES,
                        help='Number of classes (including background).')
    parser.add_argument('--backbone', type=str, default="resnext101_32x8d",
                        choices=["resnext101_32x8d", "resnet50"],
                        help='Type of backbone to use.')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use pretrained weights for the backbone.')
    parser.add_argument('--output', type=str, default=MODEL_SAVE_PATH,
                        help='Path to save the trained model.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from a checkpoint file.')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Value for gradient clipping norm (0 to disable).')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'],
                        help='Optimizer to use for training (e.g., "sgd", "adamw").')
    
    args = parser.parse_args()
    run_train(args)

if __name__ == "__main__":
    main()