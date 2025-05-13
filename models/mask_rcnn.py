import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNeXt101_32X8D_Weights 
# from torchvision.models.resnet import resnext101_32x8d, ResNeXt101_32X8D_Weights # Alternative if above is not found
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
from config import *

class MaskRCNNModel:

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, backbone_name="resnext101_32x8d", 
            clip_grad_norm=0.0, optimizer_type='sgd'):
        """
        Initializes the Mask R-CNN model, typically with a ResNeXt-101-FPN backbone.
        
        Args:
            num_classes (int): Number of classes (background + object classes).
            pretrained (bool): Whether to use pretrained weights for the backbone.
            backbone_name (str): Name of the backbone to use (default: "resnext101_32x8d").
                                 Supports "resnet50" as a fallback.
            clip_grad_norm (float): Value for gradient clipping (0 to disable).
            optimizer_type (str): Type of optimizer ('sgd' or 'adamw').
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.clip_grad_norm = clip_grad_norm
        self.optimizer_type = optimizer_type
        
        if backbone_name == "resnext101_32x8d":
            print("Using ResNeXt-101-FPN backbone for Mask R-CNN")
            
            backbone_weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if pretrained else None
            if pretrained:
                print("Loading ImageNet pretrained weights for backbone.")
            else:
                print("Initializing backbone from scratch.")
            
            backbone = resnet_fpn_backbone(
                backbone_name='resnext101_32x8d', # Explicitly use string for backbone_name
                weights=backbone_weights,
                trainable_layers=5  # All layers are trainable
            )
            
            self.model = MaskRCNN(
                backbone=backbone,
                num_classes=num_classes,
                min_size=800,       # Default TorchVision M RCNN min_size
                max_size=1333,      # Default TorchVision M RCNN max_size
                image_mean=[0.485, 0.456, 0.406], # ImageNet means
                image_std=[0.229, 0.224, 0.225]   # ImageNet stds
            )
            
        else: # Fallback to standard ResNet-50-FPN
            print(f"Backbone '{backbone_name}' not 'resnext101_32x8d', using standard ResNet-50-FPN.")
            
            model_weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
            self.model = maskrcnn_resnet50_fpn(weights=model_weights, weights_backbone=ResNet50_Weights.IMAGENET1K_V1 if pretrained and model_weights is None else None)
            # If model_weights is None (i.e., not using COCO pretrained MaskRCNN), then we might want to explicitly set backbone weights.
            # If model_weights is DEFAULT, it already includes a pretrained backbone.

            # The number of output classes for the FasterRCNN box predictor needs to be adjusted
            in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

            # The number of output classes for the MaskRCNN mask predictor needs to be adjusted
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer_mask = 256 # Default hidden layer size
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer_mask, num_classes)


        # For both custom and standard backbone, ensure the final layers match num_classes
        # Get the number of input features for the classifier
        in_features_classifier = self.model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one (e.g., for binary classification: background + object_in_hand)
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features_classifier, num_classes)
        
        # Get the number of input features for the mask classifier
        in_features_mask_predictor = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer_dim = 256 # Standard hidden layer dimension for mask predictor
        # Replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask_predictor, hidden_layer_dim, num_classes)
        
        self.model.to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) # Count only trainable parameters
        print(f"Total trainable parameters: {num_params:,}")

    def train(self, train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, model_save_path=MODEL_SAVE_PATH):
        """
        Trains the Mask R-CNN model.
        
        Args:
            train_loader: DataLoader for the training data.
            val_loader: DataLoader for the validation data.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Initial learning rate.
            model_save_path (str): Path to save the best and final model.
                
        Returns:
            dict: Dictionary containing training history (losses, metrics).
        """
        writer = SummaryWriter(TENSORBOARD_DIR) # Initialize TensorBoard writer
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0001)
            print(f"Using AdamW optimizer with learning rate {learning_rate}")
        else: # Default to SGD
            optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
            print(f"Using SGD optimizer with learning rate {learning_rate}")
        
        # Learning rate scheduler (ReduceLROnPlateau)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3 # 'verbose' argument removed or set to False
        )
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_miou": []
        }
        
        best_val_miou = 0.0
        
        for epoch in range(num_epochs):
            self.model.train() # Set model to training mode
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            running_loss_accum = 0.0 # Accumulator for average loss display
            epoch_losses = [] # List to store losses for each batch in the epoch
            
            train_pbar = tqdm(train_loader, desc=f"Training [{epoch+1}/{num_epochs}]")
            
            for i, (images, targets) in enumerate(train_pbar):
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                
                # Skip batch if any target has no bounding boxes (can cause errors in model)
                valid_batch = all(t["boxes"].numel() > 0 for t in targets)
                if not valid_batch:
                    # print(f"Skipping batch {i} due to empty boxes in one or more targets.")
                    continue
                
                optimizer.zero_grad()
                
                try:
                    loss_dict = self.model(images, targets) # Forward pass
                    
                    if not isinstance(loss_dict, dict):
                        print(f"Warning: loss_dict is not a dictionary but {type(loss_dict)}, skipping batch.")
                        continue
                    
                    losses = sum(loss for loss in loss_dict.values()) # Calculate total loss
                    
                    if not torch.isfinite(losses):
                        print(f"Warning: Loss is {losses.item()}, skipping this batch.")
                        continue
                    
                    losses.backward() # Backward pass
                    
                    if self.clip_grad_norm > 0: # Apply gradient clipping if specified
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    optimizer.step() # Update weights
                    
                    running_loss_accum += losses.item()
                    epoch_losses.append(losses.item())
                    train_pbar.set_postfix(loss=f"{running_loss_accum/(i+1):.4f}") # Update progress bar
                except Exception as e:
                    print(f"Error during training batch: {str(e)}. Skipping batch.")
                    continue # Skip to the next batch
            
            avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            history["train_loss"].append(avg_train_loss)
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            
            # Validation phase
            val_metrics = self.evaluate(val_loader, epoch=epoch, writer=writer)
            val_loss = val_metrics["loss"]
            val_miou = val_metrics["miou"]
            
            lr_scheduler.step(val_loss) # Adjust learning rate based on validation loss
            
            history["val_loss"].append(val_loss)
            history["val_miou"].append(val_miou)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")
            
            if val_miou > best_val_miou: # Save model if it's the best so far
                best_val_miou = val_miou
                self.save(model_save_path)
                print(f"Saved best model with mIoU: {val_miou:.4f} at {model_save_path}")
        
        # Save the final model (could be different from the best if mIoU fluctuates)
        final_model_path = model_save_path.replace(".pth", "_final.pth")
        self.save(final_model_path)
        print(f"Saved final model at {final_model_path}")
        
        writer.close()
        return history

    def evaluate(self, data_loader, epoch=None, writer=None):
        """
        Evaluates the model on validation/test data, calculating loss and mIoU.
        
        Args:
            data_loader: DataLoader for the validation/test data.
            epoch (int, optional): Current epoch number (for logging).
            writer (SummaryWriter, optional): TensorBoard SummaryWriter for logging.
                
        Returns:
            dict: Dictionary with evaluation metrics ('loss', 'miou').
        """
        self.model.eval() # Set model to evaluation mode
        
        all_batch_losses = []
        all_preds_for_miou = []
        all_targets_for_miou = []
        
        with torch.no_grad(): # Disable gradient calculations during evaluation
            for images, targets in tqdm(data_loader, desc="Validating"):
                images = [image.to(self.device) for image in images]
                # Targets for loss calculation need to be on device
                targets_for_loss_calc = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                
                # Ensure valid targets for loss calculation
                valid_batch_for_loss = all(t["boxes"].numel() > 0 for t in targets_for_loss_calc)

                current_batch_loss = 0.0
                try:
                    # 1. Calculate loss (temporarily switch to train mode for model to return loss_dict)
                    if valid_batch_for_loss:
                        self.model.train() 
                        loss_dict = self.model(images, targets_for_loss_calc)
                        self.model.eval() # Switch back to eval mode
                        
                        if isinstance(loss_dict, dict):
                            losses = sum(loss for loss in loss_dict.values())
                            current_batch_loss = losses.item()
                            all_batch_losses.append(current_batch_loss)
                        else:
                            print(f"Unexpected output type during loss calculation: {type(loss_dict)}")
                    # else: if not valid_batch_for_loss, loss for this batch is effectively 0 or skipped.

                    # 2. Get predictions for mIoU calculation (model should be in eval mode)
                    outputs = self.model(images) # Get predictions
                    
                    for output, target_cpu in zip(outputs, targets): # Use original targets (on CPU) for mIoU
                        if not isinstance(output, dict):
                            print(f"Error: prediction output is not a dictionary but {type(output)}")
                            continue
                            
                        pred_boxes = output['boxes'].cpu()
                        pred_scores = output['scores'].cpu()
                        pred_labels = output['labels'].cpu()
                        pred_masks = output['masks'].cpu() # Shape: (N_preds, 1, H, W)
                        
                        # Filter predictions by score for mIoU calculation
                        # Using a fixed or passed threshold might be better than hardcoded 0.5 here
                        # For now, assume predict() handles thresholding if this is for general eval
                        keep = pred_scores > 0.05 # Lower threshold for mIoU calculation to include more preds
                        
                        # Store predictions and ground truth for mIoU calculation
                        all_preds_for_miou.append({
                            'boxes': pred_boxes[keep],
                            'scores': pred_scores[keep],
                            'labels': pred_labels[keep],
                            'masks': pred_masks[keep] # Store filtered masks
                        })
                        
                        all_targets_for_miou.append({ # Ensure GT masks are in the expected format for calculate_miou
                            'boxes': target_cpu['boxes'],
                            'labels': target_cpu['labels'],
                            'masks': target_cpu['masks'] # GT masks: (N_gt, H, W)
                        })
                except Exception as e:
                    print(f"Error during evaluation batch: {str(e)}. Skipping batch metrics.")
                    continue
        
        avg_loss = sum(all_batch_losses) / len(all_batch_losses) if all_batch_losses else 0.0
        
        miou = 0.0
        if all_preds_for_miou and all_targets_for_miou:
            try:
                miou = self.calculate_miou(all_preds_for_miou, all_targets_for_miou)
            except Exception as e:
                print(f"Error calculating mIoU: {str(e)}")
                miou = 0.0 
        else:
            print("No valid predictions or targets to calculate mIoU.")
            
        if writer is not None and epoch is not None:
            writer.add_scalar('Loss/val', avg_loss, epoch)
            writer.add_scalar('mIoU/val', miou, epoch)
        
        metrics = {"loss": avg_loss, "miou": miou}
        print(f"Validation - Avg Loss: {avg_loss:.4f}, mIoU: {miou:.4f}")
        return metrics

    def calculate_miou(self, all_preds, all_targets, iou_threshold=0.5, mask_binarize_threshold=0.5):
        """
        Calculates Mean Intersection over Union (mIoU) for masks.
        This version finds the *best match* for each GT mask.

        Args:
            all_preds (list): List of dictionaries with predictions.
                              Each dict contains 'masks': Tensor (N_preds, 1, H, W) float [0,1].
            all_targets (list): List of dictionaries with ground truths.
                                Each dict contains 'masks': Tensor (N_gt, H, W) uint8/bool.
            iou_threshold (float): IoU threshold for considering a match (not strictly used for pure mIoU per GT).
            mask_binarize_threshold (float): Threshold to convert predicted float masks to binary.

        Returns:
            float: Mean IoU averaged over all ground truth masks.
        """
        all_ious_for_each_gt_mask = [] # Stores the best IoU found for each GT mask

        for pred_dict, target_dict in zip(all_preds, all_targets):
            pred_masks_tensor = pred_dict.get('masks', torch.empty(0, 1, 0, 0)) # (N_preds, 1, H, W)
            gt_masks_tensor = target_dict.get('masks', torch.empty(0, 0, 0))     # (N_gt, H, W)

            # --- Convert to NumPy and Binarize ---
            # Predictions: Binarize and convert to numpy. Squeeze the channel dimension.
            pred_masks_np = (pred_masks_tensor.cpu().numpy() > mask_binarize_threshold).astype(np.uint8)
            if pred_masks_np.ndim == 4 and pred_masks_np.shape[1] == 1:
                pred_masks_np = pred_masks_np[:, 0, :, :] # -> shape (N_preds, H, W)
            
            # Ground Truth: Convert to numpy. Ensure it's (N_gt, H, W).
            gt_masks_np = gt_masks_tensor.cpu().numpy().astype(np.uint8)
            if gt_masks_np.ndim == 4 and gt_masks_np.shape[1] == 1: # Handle if GT also has channel dim
                gt_masks_np = gt_masks_np[:, 0, :, :]


            num_preds = pred_masks_np.shape[0]
            num_gts = gt_masks_np.shape[0]

            if num_gts == 0:
                continue # No GT masks in this image, skip.

            if num_preds == 0:
                # If no predictions, IoU for all GTs in this image is 0.
                all_ious_for_each_gt_mask.extend([0.0] * num_gts)
                continue

            # --- Check H, W dimensions (after handling channels) ---
            # This assumes all masks in a batch have same H,W, which should be true after dataloader
            pred_h, pred_w = pred_masks_np.shape[1:] if num_preds > 0 else (0,0)
            gt_h, gt_w = gt_masks_np.shape[1:] if num_gts > 0 else (0,0)

            if pred_h != gt_h or pred_w != gt_w :
                 if num_preds > 0 and num_gts > 0 : # Only print if both exist but mismatch
                    print(f"WARNING: H,W dimensions mismatch! Pred: ({pred_h},{pred_w}), GT: ({gt_h},{gt_w}). Cannot calculate IoU for this image.")
                 all_ious_for_each_gt_mask.extend([0.0] * num_gts)
                 continue

            iou_matrix = np.zeros((num_gts, num_preds))

            for i in range(num_gts): # Iterate over GT masks
                gt_mask = gt_masks_np[i] # Shape (H, W)
                if gt_mask.sum() == 0: continue # Skip empty GT masks

                for j in range(num_preds): # Iterate over predicted masks
                    pred_mask = pred_masks_np[j] # Shape (H, W)
                    if pred_mask.sum() == 0: continue # Skip empty predicted masks
                    
                    intersection = np.logical_and(gt_mask, pred_mask).sum()
                    union = np.logical_or(gt_mask, pred_mask).sum()

                    if union > 0:
                        iou_matrix[i, j] = intersection / union
            
            # For each GT mask, find the maximum IoU with any predicted mask
            if num_preds > 0:
                max_iou_per_gt = iou_matrix.max(axis=1) 
            else: # Should have been caught by num_preds == 0 check earlier
                max_iou_per_gt = np.zeros(num_gts)
            
            all_ious_for_each_gt_mask.extend(max_iou_per_gt.tolist())

        if not all_ious_for_each_gt_mask:
            print("Warning: calculate_miou: No IoUs calculated (e.g., no GT masks found). Returning 0.0.")
            return 0.0

        miou = np.mean(all_ious_for_each_gt_mask)
        return float(miou)

    def predict(self, image, score_threshold=0.5):
        """
        Performs prediction on a single image.
        
        Args:
            image (np.ndarray or torch.Tensor): Input image.
            score_threshold (float): Confidence threshold for predictions.
            
        Returns:
            dict: Dictionary with 'masks', 'boxes', 'scores', 'labels' above the threshold.
                  Masks are binary numpy arrays (N, H, W).
        """
        self.model.eval()
        
        # --- Image Preprocessing ---
        if isinstance(image, np.ndarray):
            if image.ndim == 2: # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4: # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            if image.dtype == np.uint8:
                image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255.0)
            else: # Assuming already float and in range [0,1]
                image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float()
        elif isinstance(image, torch.Tensor):
            image_tensor = image.clone() # Work on a copy
            if image_tensor.ndim == 2: # Grayscale
                image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
            elif image_tensor.shape[0] == 1: # Single channel (likely grayscale)
                image_tensor = image_tensor.repeat(3,1,1)
            elif image_tensor.shape[0] == 4: # RGBA
                image_tensor = image_tensor[:3, :, :]
            
            if image_tensor.max() > 1.0 and image_tensor.dtype != torch.uint8 : # If float but not in [0,1]
                image_tensor = image_tensor.div(255.0)
            elif image_tensor.dtype == torch.uint8:
                 image_tensor = image_tensor.float().div(255.0)

        else:
            raise TypeError(f"Unsupported image format: {type(image)}")

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0) # Add batch dimension
        
        image_tensor = image_tensor.to(self.device)
        
        empty_result = lambda: {'masks': np.array([]), 'boxes': np.array([]), 'scores': np.array([]), 'labels': np.array([])}

        try:
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            if not predictions or len(predictions) == 0:
                return empty_result()
            
            pred = predictions[0] # Results for the first (and only) image in the batch
            
            scores = pred['scores'].cpu().numpy()
            keep_indices = scores > score_threshold
            
            masks_raw = pred['masks'].cpu().numpy()[keep_indices] # (N_kept, 1, H, W)
            boxes = pred['boxes'].cpu().numpy()[keep_indices]
            final_scores = scores[keep_indices]
            labels = pred['labels'].cpu().numpy()[keep_indices]
            
            binary_masks = np.array([])
            if masks_raw.size > 0:
                # Binarize masks (threshold > 0.5) and squeeze channel dimension
                binary_masks = (masks_raw > 0.5).astype(np.uint8).squeeze(axis=1) # (N_kept, H, W)
            
            return {
                'masks': binary_masks,
                'boxes': boxes,
                'scores': final_scores,
                'labels': labels
            }
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return empty_result()

    def save(self, path):
        """
        Saves the model's state_dict to a file.
        
        Args:
            path (str): Path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True) # Create directory if it doesn't exist
        
        # Handle saving models wrapped with DataParallel
        model_state_dict = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()
        torch.save(model_state_dict, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """
        Loads model weights from a file.
        
        Args:
            path (str): Path to the model file.
        
        Returns:
            bool: True if loading was successful, False otherwise.
        """
        if not os.path.exists(path):
            print(f"Warning: Model file {path} not found.")
            return False
        
        try:
            state_dict = torch.load(path, map_location=self.device) 
            
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
                
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False