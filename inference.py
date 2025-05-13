import os
import glob
import torch
import cv2
import numpy as np
from tqdm import tqdm
import random
import time
import argparse
from config import * 
from models.mask_rcnn import MaskRCNNModel

def get_class_names(num_classes_from_args):
    """
    Gets the list of class names for inference.
    Returns a list with 'background' at index 0.

    Args:
        num_classes_from_args (int): Total number of classes (incl. background) from the model/arguments.

    Returns:
        list: List of class names, with 'background' at index 0.
    """
    class_names = None
    expected_foreground_classes = num_classes_from_args - 1

    # 1. Try with CLASS_NAMES from config.py
    if 'CLASS_NAMES' in globals():
        config_class_names = globals()['CLASS_NAMES']
        if isinstance(config_class_names, (list, tuple)):
            print("INFO: Attempting to use CLASS_NAMES from config.py.")
            # We assume config.CLASS_NAMES may or may not include 'background'.
            # The safest check is against the number of foreground classes.
            if len(config_class_names) == expected_foreground_classes:
                 # config.py contains only foreground names
                 print(f"INFO: Using {len(config_class_names)} foreground names from config.py.")
                 class_names = ["background"] + list(config_class_names)
            elif len(config_class_names) == num_classes_from_args:
                 # config.py already includes 'background'
                 print(f"INFO: Using {len(config_class_names)} names (including background) from config.py.")
                 class_names = list(config_class_names)
            else:
                print(f"WARNING: CLASS_NAMES in config.py has {len(config_class_names)} elements, but {expected_foreground_classes} (foreground) or {num_classes_from_args} (total) were expected. Ignoring.")
        else:
             print("WARNING: CLASS_NAMES found in config.py but it's not a list/tuple. Ignoring.")

    # 2. Fallback: Generate default names
    if class_names is None:
        print(f"INFO: Generating default class names ('Obj 1', ..., 'Obj {expected_foreground_classes}').")
        foreground_names = [f"Obj {i}" for i in range(1, num_classes_from_args)]
        class_names = ["background"] + foreground_names

    # Final safety check on length
    if len(class_names) != num_classes_from_args:
         print(f"INTERNAL ERROR: Final class_names length ({len(class_names)}) != num_classes ({num_classes_from_args}). Regenerating.")
         foreground_names = [f"Obj {i}" for i in range(1, num_classes_from_args)]
         class_names = ["background"] + foreground_names

    # Ensure the first element is 'background' (case-insensitive for robustness)
    if not class_names[0].lower() == "background":
         print(f"WARNING: The first class name '{class_names[0]}' is not 'background'. This might cause indexing issues.")
         # One could force it: class_names[0] = "background"

    print(f"Final class names (used for output): {class_names}")
    return class_names

def get_colors(num_classes):
    """Generates distinct random colors for each class (excluding background)."""
    # Use RANDOM_SEED from config if imported, otherwise a default
    seed = globals().get('RANDOM_SEED', 42)
    random.seed(seed)
    colors = [(0, 0, 0)] # Black for background
    for i in range(1, num_classes):
        # Generate brighter colors
        colors.append((random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)))
    return colors

def draw_predictions(image, predictions, class_names, colors, threshold):
    """
    Draws masks, bounding boxes, labels, and scores on the image.
    Robust function to handle edges and potential errors.

    Args:
        image (np.ndarray): Original image (BGR).
        predictions (dict): Output from model.predict(). Contains 'masks', 'boxes', 'scores', 'labels'.
        class_names (list): List of class names (index 0 = background).
        colors (list): List of BGR color tuples (index 0 = background).
        threshold (float): Confidence threshold used for prediction.

    Returns:
        np.ndarray: Image with annotations.
    """
    output_image = image.copy()
    masks = predictions.get('masks')     # Shape (N, H, W), uint8 or None
    boxes = predictions.get('boxes')     # Shape (N, 4), float32 or None
    scores = predictions.get('scores')   # Shape (N,), float32 or None
    labels = predictions.get('labels')   # Shape (N,), int64 or None

    # Verify that all necessary predictions are present and not empty
    if masks is None or boxes is None or scores is None or labels is None or len(scores) == 0:
        # print("No valid predictions found to draw.")
        return output_image # Return original image if there's nothing to draw

    overlay = output_image.copy()
    # Use MASK_ALPHA from config if it exists, otherwise default
    alpha = globals().get('MASK_ALPHA', 0.5)
    num_predictions = len(scores)

    for i in range(num_predictions):
        # Only instances above the threshold (though predict should already filter)
        if scores[i] < threshold:
            continue

        # --- Data Extraction ---
        box = boxes[i].astype(np.int32)
        label_idx = labels[i]
        score = scores[i]
        mask = masks[i] # Shape (H, W)

        # --- Label Index Validation ---
        if label_idx <= 0 or label_idx >= len(class_names):
            print(f"  WARNING: Invalid label index {label_idx}. Max: {len(class_names)-1}. Using 'Unknown'.")
            class_name = "Unknown"
            # Use a fallback color (e.g., white) if the index is out of range for colors too
            color = colors[0] if label_idx >= len(colors) else colors[label_idx] if label_idx >=0 else (255,255,255)
            if label_idx >= len(colors) or label_idx < 0: color = (255, 255, 255)
        else:
            class_name = class_names[label_idx]
            color = colors[label_idx]

        # --- Draw Mask (with shape check) ---
        try:
            if mask.shape == overlay.shape[:2]: # Checks H, W
                 overlay[mask > 0] = color
            else:
                 print(f"  WARNING: Mask shape {mask.shape} does not match overlay {overlay.shape[:2]}. Mask not drawn.")
                 # You could attempt a resize here, but it's better if predictions are correct
                 # mask_resized = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
                 # overlay[mask_resized > 0] = color
        except IndexError:
             print(f"  WARNING: Index error while drawing mask {i}.")
             continue # Skip this instance

        # --- Draw Bounding Box (with clipping to edges) ---
        x1, y1, x2, y2 = box
        h, w = output_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x1 >= x2 or y1 >= y2: # Skip invalid/degenerate boxes
            print(f"  WARNING: Box {i} invalid after clipping: ({x1},{y1},{x2},{y2}).")
            continue
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2) # Thickness 2

        # --- Draw Text (Label + Score, with edge handling) ---
        text = f"{class_name}: {score:.2f}"
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, font_thickness)

        # Position above the box by default
        text_x = x1 + 2
        text_y = y1 - 5 # Position of the text baseline

        # If too close to the top edge, move below
        if text_y - text_height - baseline < 0:
            text_y = y2 + text_height + 5 # Below the box

        # If it goes off the bottom edge (unlikely if moved below), bring it back in
        if text_y > h:
             text_y = y1 + text_height + 5 # Inside the box, at the top

        # Background for readability
        cv2.rectangle(output_image,
                      (text_x, text_y - text_height - baseline), # Top-left of the background
                      (text_x + text_width, text_y + baseline),  # Bottom-right of the background
                      color, -1) # Filled
        # Text (black for contrast)
        cv2.putText(output_image, text, (text_x, text_y),
                    font_face, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)

    # Apply the mask overlay with transparency to the final image
    cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)

    return output_image

# --- Main Inference Function ---

def run_inference(args):
    """
    Runs inference using the specified Mask R-CNN model on an image
    or a directory of images.

    Args:
        args: Object (similar to argparse) containing necessary attributes:
              - model (str): Path to the .pth model file.
              - input (str): Path to the input image or directory.
              - output (str): Directory to save results.
              - threshold (float): Confidence threshold for predictions.
              - num_classes (int): Total number of classes (incl. background).
              - backbone (str): Name of the backbone used (e.g., 'resnext101_32x8d').
              - show (bool): Whether to show the resulting image (only for single input).
              - class_names (str or None): (Optional) Comma-separated string of class names.
                                            If not provided or None, names from config.py
                                            or default generated names will be used.
    """
    start_time = time.time()
    print("\n--- Starting Inference ---")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Confidence threshold: {args.threshold}")

    num_classes_to_use = args.num_classes
    print(f"Number of Classes (incl. background): {num_classes_to_use}")
    print(f"Backbone: {args.backbone}")
    if hasattr(args, 'class_names') and args.class_names: # Print only if the attribute exists and is set
        print(f"Externally provided class names: {args.class_names}")
    if args.show:
        print("Display mode: Active (only for single image)")

    # --- 1. Model Preparation ---
    print("Initializing model structure...")
    try:
        # Pass only num_classes and backbone; weights will be loaded later
        model_wrapper = MaskRCNNModel(
            num_classes=num_classes_to_use,
            pretrained=False, # We are not using torchvision pretrained weights here
            backbone_name=args.backbone
        )
    except Exception as e:
         print(f"ERROR during MaskRCNNModel initialization: {e}")
         print("Check if num_classes and backbone_name are correct.")
         return # Exit if model cannot be created

    # Load trained weights from the specified path
    print(f"Loading weights from: {args.model}")
    if not model_wrapper.load(args.model):
        print(f"ERROR: Could not load model weights from {args.model}. Exiting.")
        return

    # Set the model to evaluation mode (important!)
    model_wrapper.model.eval()
    print("Model loaded successfully and set to evaluation mode.")

    # --- 2. Prepare Class Names and Colors ---
    # get_class_names now uses globals() to check for CLASS_NAMES in config
    class_names = get_class_names(num_classes_to_use)
    colors = get_colors(num_classes_to_use)

    # --- 3. Create Output Directory ---
    try:
        os.makedirs(args.output, exist_ok=True)
        print(f"Results will be saved in: {args.output}")
    except OSError as e:
        print(f"ERROR: Could not create output directory '{args.output}': {e}")
        return

    # --- 4. Identify Input (File or Directory) ---
    image_paths = []
    process_multiple = False
    if os.path.isdir(args.input):
        print(f"Input is a directory. Searching for images...")
        supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
        for ext in supported_extensions:
            # Search recursively if needed? Not for now.
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
        if not image_paths:
            print(f"WARNING: No images with supported extensions found in: {args.input}")
            return
        image_paths.sort() # Sort for more predictable output
        print(f"Found {len(image_paths)} images.")
        process_multiple = True
    elif os.path.isfile(args.input):
        print(f"Input is a single file: {args.input}")
        image_paths.append(args.input)
        process_multiple = False
    else:
        print(f"ERROR: Input path '{args.input}' is not a valid file or directory.")
        return

    # --- 5. Inference Loop over Images ---
    processed_count = 0
    errors_count = 0
    total_instances_detected = 0

    # Disable tqdm if processing only one image
    iterable = tqdm(image_paths, desc="Inference", disable=not process_multiple)

    for img_path in iterable:
        base_filename = os.path.basename(img_path)
        # Add a clear suffix to the output filename
        output_filename = f"{os.path.splitext(base_filename)[0]}_inference.png" # Use PNG for lossless output
        output_path = os.path.join(args.output, output_filename)

        if not process_multiple: # Additional log for single image
             print(f"\nProcessing: {base_filename}")

        # Load image (BGR for OpenCV)
        try:
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                tqdm.write(f"WARNING: Could not load image {img_path}. Skipped.")
                errors_count += 1
                continue
        except Exception as e:
            tqdm.write(f"ERROR while loading {img_path}: {e}. Skipped.")
            errors_count += 1
            continue

        # Perform prediction (MaskRCNNModel.predict handles conversion/normalization)
        try:
            prediction_start_time = time.time()
            # Pass BGR image, predict should handle it
            predictions = model_wrapper.predict(image_bgr, score_threshold=args.threshold)
            prediction_time = time.time() - prediction_start_time
            num_detected = len(predictions.get('scores', []))
            total_instances_detected += num_detected
            if not process_multiple:
                 print(f"  Prediction completed in {prediction_time:.3f}s. Detected {num_detected} instances.")

        except Exception as e:
             tqdm.write(f"ERROR during prediction on {img_path}: {e}. Skipped.")
             import traceback
             tqdm.write(traceback.format_exc()) # Print traceback for debugging
             errors_count += 1
             continue

        # Draw results on the BGR image
        try:
            drawing_start_time = time.time()
            output_image = draw_predictions(image_bgr, predictions, class_names, colors, args.threshold)
            drawing_time = time.time() - drawing_start_time
            if not process_multiple:
                print(f"  Drawing completed in {drawing_time:.3f}s.")

        except Exception as e:
             tqdm.write(f"ERROR while drawing predictions on {img_path}: {e}. Skipped.")
             errors_count += 1
             continue

        # Save the resulting image (PNG)
        try:
            success = cv2.imwrite(output_path, output_image)
            if not success:
                 raise IOError("imwrite returned False")
            processed_count += 1
        except Exception as e:
            tqdm.write(f"ERROR while saving {output_path}: {e}. Skipped.")
            errors_count += 1
            continue

        # Show the image if requested (only for single image)
        if args.show and not process_multiple:
            print("Displaying image... Press any key on the window to close.")
            try:
                # Resize window if image too large (optional)
                max_display_h = 800
                h_img, w_img = output_image.shape[:2]
                if h_img > max_display_h:
                    scale_factor = max_display_h / float(h_img)
                    display_width = int(w_img * scale_factor)
                    display_height = max_display_h
                    display_img = cv2.resize(output_image, (display_width, display_height), interpolation=cv2.INTER_AREA)
                else:
                    display_img = output_image

                cv2.imshow(f"Inference Result - {base_filename}", display_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error during image display: {e}")

    # --- 6. Final Summary ---
    end_time = time.time()
    total_time = end_time - start_time
    print("\n--- Inference Completed ---")
    print(f"Total images found: {len(image_paths)}")
    print(f"Images processed successfully: {processed_count}")
    if errors_count > 0:
        print(f"Errors / Skipped images: {errors_count}")
    print(f"Total instances detected (above threshold {args.threshold}): {total_instances_detected}")
    print(f"Results (annotated images) saved in: {args.output}")
    print(f"Total time taken: {total_time:.2f} seconds")
    if processed_count > 0:
         avg_time = total_time / processed_count
         print(f"Average time per image (I/O, pred, draw): {avg_time:.3f} seconds ({1/avg_time:.2f} FPS)")
    print("--------------------------\n")


# --- Block for standalone testing ---
if __name__ == '__main__':
    print("Running inference.py as a standalone script for testing.")

    # Define a basic parser for testing this function
    parser = argparse.ArgumentParser(description="Test Script for run_inference")

    # Required arguments
    parser.add_argument('--model', type=str, required=True, help='Path to the .pth model file')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image or image directory')
    parser.add_argument('--output', type=str, default=RESULTS_DIR, help=f"Output directory for results (default: {RESULTS_DIR})")
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for predictions (default: 0.5)')
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, help=f"Number of classes (incl. background) (default: {NUM_CLASSES})")
    parser.add_argument('--backbone', type=str, default="resnext101_32x8d", choices=["resnext101_32x8d", "resnet50"], help='Backbone type used in the model (default: resnext101_32x8d)')
    parser.add_argument('--show', action='store_true', help='Show processed images (only if input is a single file)')
    # Optional argument for class names, matching how run_inference might receive it
    parser.add_argument('--class_names', type=str, default=None, help='(Optional) Comma-separated string of class names. If not provided, uses config or generates defaults.')

    test_args = parser.parse_args()
    run_inference(test_args)