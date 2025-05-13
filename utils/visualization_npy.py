import os
import sys
import numpy as np
import cv2
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import * 

def visualize_samples(dataset_path, output_dir, num_images=5):
    """
    Visualizes and saves the first N images from the dataset with overlaid masks.
    
    Args:
        dataset_path: Path to the .npy dataset file.
        output_dir: Directory to save the visualizations.
        num_images: Number of images to visualize (the first N).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}...")
    
    try:
        data = np.load(dataset_path, allow_pickle=True)
        print(f"Dataset loaded, total size: {len(data)} samples.")
        
        if num_images > len(data):
            num_images = len(data)
            print(f"Number of images to display reduced to {num_images} (dataset size).")
        
        data_to_visualize = data[:num_images] # Process only the first N samples
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Visualize each sample
    for i, sample in enumerate(tqdm(data_to_visualize, desc="Visualizing images")):
        image_path = sample["image_path"]
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB for consistent display
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            continue
        
        mask = sample["mask"]
        
        # Ensure the mask has the correct dimensions, resize if necessary
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        # --- Create a combined visualization: original image, mask, overlay ---
        
        # 1. Original image
        orig_image = image.copy()
        
        # 2. Colored mask (for better visualization)
        colored_mask_viz = np.zeros_like(image)
        colored_mask_viz[mask > 0] = [0, 0, 255]  # Blue color for the mask
        
        # 3. Image with overlaid mask
        masked_image = image.copy()
        alpha = 0.5  # Mask opacity
        mask_bool = mask > 0
        # Apply color only where mask is True
        masked_image[mask_bool] = (masked_image[mask_bool] * (1 - alpha) + 
                                   colored_mask_viz[mask_bool] * alpha).astype(np.uint8)
        
        # Create a composite image with all three visualizations side-by-side
        composite = np.concatenate([orig_image, colored_mask_viz, masked_image], axis=1)
        
        # Add information as text to the composite image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White
        font_thickness = 1
        
        # Information to display
        clip_name = sample.get("clip_name", "N/A")
        frame_id = sample.get("frame_id", "N/A")
        camera_id = sample.get("camera_id", "N/A")
        
        info_text = f"Clip: {clip_name}, Frame: {frame_id}, Cam: {camera_id}"
        
        # Add text to the bottom of the image
        (text_w, text_h), baseline = cv2.getTextSize(info_text, font, font_scale, font_thickness)
        text_x = 10
        text_y = composite.shape[0] - 10  # 10 pixels from the bottom
        
        # Add a black rectangle behind the text for readability
        cv2.rectangle(composite, 
                     (text_x - 5, text_y - text_h - 5), 
                     (text_x + text_w + 5, text_y + baseline - 2), # Adjusted for better fit
                     (0, 0, 0), 
                     -1)  # -1 to fill the rectangle
        
        cv2.putText(composite, 
                   info_text, 
                   (text_x, text_y - (baseline//2)), # Adjust y for baseline
                   font, 
                   font_scale, 
                   font_color, 
                   font_thickness)
        
        # Add headers to clarify each image segment
        headers = ["Original Image", "Mask", "Overlay"]
        segment_width = composite.shape[1] // 3
        
        for idx, header_text in enumerate(headers):
            (text_w_h, text_h_h), baseline_h = cv2.getTextSize(header_text, font, font_scale, font_thickness)
            header_x = idx * segment_width + (segment_width - text_w_h) // 2
            header_y = 20  # 20 pixels from the top
            
            # Add black rectangle for header background
            cv2.rectangle(composite, 
                         (header_x - 5, header_y - text_h_h - 5), 
                         (header_x + text_w_h + 5, header_y + baseline_h - 2), 
                         (0, 0, 0), 
                         -1)
            
            cv2.putText(composite, 
                       header_text, 
                       (header_x, header_y - (baseline_h//2)), 
                       font, 
                       font_scale, 
                       font_color, 
                       font_thickness)
        
        # Save the composite image
        output_file = os.path.join(output_dir, f"sample_{i+1:03d}.jpg")
        # Convert back to BGR for OpenCV imwrite if image was RGB
        cv2.imwrite(output_file, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)) 
    
    print(f"Visualizations saved to {output_dir}")

def main():
    """
    Script to visualize the first N images from the training dataset
    and overlay the in-hand object mask.
    """
    parser = argparse.ArgumentParser(description='Visualizes the first N images from the dataset with masks.')
    parser.add_argument('--dataset', choices=['train', 'val', 'test'], default='train',
                        help='Dataset split to visualize (train, val, or test).')
    parser.add_argument('--images', type=int, default=5,
                        help='Number of images to visualize (the first N from the selected dataset).')
    parser.add_argument('--output_dir', default='visualizations',
                        help='Base directory to save the visualization images.')
    
    args = parser.parse_args()
    
    # Determine the dataset path based on the argument
    if args.dataset == 'train':
        dataset_path = TRAIN_CACHE_PATH
    elif args.dataset == 'val':
        dataset_path = VAL_CACHE_PATH
    else: # 'test'
        dataset_path = TEST_CACHE_PATH
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset cache file not found at {dataset_path}")
        print("Please ensure the dataset has been preprocessed and cached.")
        return

    # Create the full output directory path (e.g., visualizations/train)
    full_output_dir = os.path.join(args.output_dir, args.dataset)
    
    # Visualize the samples
    visualize_samples(dataset_path, full_output_dir, args.images)

if __name__ == "__main__":
    main()