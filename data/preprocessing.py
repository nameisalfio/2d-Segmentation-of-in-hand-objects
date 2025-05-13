import os
import json
import numpy as np
import cv2
import glob
from tqdm import tqdm
import sys
from pycocotools import mask as mask_utils 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import * 


def process_frame(image_info, annotations, categories, dataset_path, debug_dir=None):
    """
    Processes a single frame from the VISOR dataset, extracting in-hand objects.
    """
    image_id = image_info["id"]
    image_path = os.path.join(dataset_path, "images", image_info["file_name"])
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB
    height, width = image.shape[:2]
    
    # Filter annotations for this specific image
    image_annotations = [ann for ann in annotations if ann["image_id"] == image_id]
    
    # Get category IDs for "hand" and other objects
    hand_category_id = None
    object_category_id = None # Assuming only one other object category for "object_in_hand"
    for cat in categories:
        if "hand" in cat["name"].lower(): # Case-insensitive check for "hand"
            hand_category_id = cat["id"]
        else: # Assuming the other category is the object of interest
            object_category_id = cat["id"]
    
    # Filter for hand annotations that are in contact with an object
    hands_in_contact = [ann for ann in image_annotations if 
                      ann["category_id"] == hand_category_id and 
                      ann.get("isincontact", 0) == 1] # isincontact=1 means hand is touching an object
    
    # Filter for object annotations
    objects = [ann for ann in image_annotations if ann["category_id"] == object_category_id]
    
    if not hands_in_contact or not objects:
        return None  # No hands in contact with objects, or no objects found in this frame
    
    # Associate hands with objects based on spatial overlap (bounding box intersection)
    objects_in_hand = []
    for hand in hands_in_contact:
        hand_box_coco = hand["bbox"] # [x, y, width, height]
        hand_x1, hand_y1, hand_w, hand_h = hand_box_coco
        hand_x2, hand_y2 = hand_x1 + hand_w, hand_y1 + hand_h
        
        for obj in objects:
            obj_box_coco = obj["bbox"] # [x, y, width, height]
            obj_x1, obj_y1, obj_w, obj_h = obj_box_coco
            obj_x2, obj_y2 = obj_x1 + obj_w, obj_y1 + obj_h
            
            # Check for bounding box overlap
            if (max(hand_x1, obj_x1) < min(hand_x2, obj_x2) and 
                max(hand_y1, obj_y1) < min(hand_y2, obj_y2)):
                # Overlap detected, consider this object as potentially in hand
                if obj not in objects_in_hand: # Add only if not already added
                    objects_in_hand.append(obj)
    
    if not objects_in_hand:
        return None  # No objects determined to be in hand in this frame after association
    
    # Create a combined mask of all objects in hand
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Lists to collect individual masks, boxes, labels, and object IDs
    individual_masks_list = []
    individual_boxes_list = []
    individual_labels_list = []
    individual_object_ids_list = []
    
    for obj_ann in objects_in_hand:
        segmentation = obj_ann.get("segmentation", [])
        
        if not segmentation:
            continue # Skip if no segmentation data
            
        # Segmentation can be in RLE format or polygon format
        current_obj_mask = None
        if isinstance(segmentation, dict):  # RLE format
            current_obj_mask = mask_utils.decode(segmentation)
        elif isinstance(segmentation, list) and len(segmentation) > 0:  # Polygon format
            current_obj_mask = np.zeros((height, width), dtype=np.uint8)
            if isinstance(segmentation[0], list):  # List of polygons (for objects with holes)
                for poly_points in segmentation:
                    poly = np.array(poly_points, np.int32).reshape((-1, 2))
                    cv2.fillPoly(current_obj_mask, [poly], 1)
            elif isinstance(segmentation[0], (int, float)):  # Single flat polygon list
                poly = np.array(segmentation, np.int32).reshape((-1, 2))
                cv2.fillPoly(current_obj_mask, [poly], 1)
        
        if current_obj_mask is None or current_obj_mask.sum() == 0: # Skip if mask is invalid or empty
            continue
            
        coco_box = obj_ann["bbox"] # COCO format: [x, y, width, height]
        x1, y1, w_box, h_box = coco_box
        # Convert to [x1, y1, x2, y2] format for easier use with some libraries
        box_xyxy = [float(x1), float(y1), float(x1 + w_box), float(y1 + h_box)]
        
        category_id = obj_ann["category_id"] # Get the category ID of the object
        
        # Add to the combined mask for all in-hand objects
        combined_mask = np.logical_or(combined_mask, current_obj_mask).astype(np.uint8)
        
        # Store individual object information
        individual_masks_list.append(current_obj_mask)
        individual_boxes_list.append(box_xyxy)
        individual_labels_list.append(category_id) # Store the object's category ID
        individual_object_ids_list.append(obj_ann["id"]) # Store the unique annotation ID for this object instance
    
    # Save debug image if requested and there are valid masks
    if debug_dir and len(individual_masks_list) > 0:
        # Create an image showing both boxes and masks for debugging
        debug_vis_path = os.path.join(debug_dir, f"image_{image_id}_annotated_objects.jpg")
        vis_image = image.copy() # Work on a copy of the original image
        
        mask_color_for_overlay = (0, 0, 255) # Blue for mask overlay
        
        for i, ind_mask in enumerate(individual_masks_list):
            # Apply semi-transparent colored masks
            mask_overlay_temp = np.zeros_like(vis_image)
            mask_bool_ind = ind_mask > 0
            mask_overlay_temp[mask_bool_ind] = mask_color_for_overlay
            
            alpha_debug = 0.4  # Transparency level for individual mask overlay
            vis_image[mask_bool_ind] = ((1-alpha_debug) * vis_image[mask_bool_ind] + 
                                      alpha_debug * mask_overlay_temp[mask_bool_ind]).astype(np.uint8)

            # Also add the corresponding bounding box
            x1_b, y1_b, x2_b, y2_b = [int(b_coord) for b_coord in individual_boxes_list[i]]
            cv2.rectangle(vis_image, (x1_b, y1_b), (x2_b, y2_b), mask_color_for_overlay, 2)
            
            # Add label with category ID
            label_id = individual_labels_list[i]
            cv2.putText(vis_image, f"Obj CatID: {label_id}", (x1_b, y1_b - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, mask_color_for_overlay, 2)
        
        cv2.imwrite(debug_vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)) # Save BGR
    
    # Create the data record for this frame
    frame_data = {
        "image_id": image_id,
        "file_name": image_info["file_name"],
        "image_path": image_path,
        "mask": combined_mask, # Combined mask of all in-hand objects
        "individual_masks": individual_masks_list, # List of masks for each in-hand object
        "boxes": individual_boxes_list, # List of [x1,y1,x2,y2] boxes for each
        "labels": individual_labels_list, # List of category_ids for each object
        "object_ids": individual_object_ids_list, # List of original annotation IDs for each object
        "image_shape": (height, width) # Original image dimensions
    }
    
    return frame_data

def process_visor_dataset(json_file_path, base_dataset_path=HOT3D_DATASET_PATH, output_dataset_type='train', debug_mode=False):
    """
    Processes the VISOR dataset from a JSON annotation file and generates an .npy cache file.
    
    Args:
        json_file_path (str): Path to the JSON annotation file.
        base_dataset_path (str): Base path of the dataset where images are stored.
        output_dataset_type (str): Type of dataset being processed ('train', 'val', or 'test'),
                                   used to determine the output .npy file name.
        debug_mode (bool): If True, saves debug images.
    
    Returns:
        str or None: Path to the processed data file (.npy), or None if processing fails or yields no data.
    """
    print(f"Processing JSON annotation file: {json_file_path}")
    
    with open(json_file_path, 'r') as f:
        annotation_data = json.load(f)
    
    images_data = annotation_data.get("images", [])
    annotations_data = annotation_data.get("annotations", [])
    categories_data = annotation_data.get("categories", [])
    
    print(f"Found {len(images_data)} images, {len(annotations_data)} annotations, and {len(categories_data)} categories.")
    print(f"Available categories: {[(cat['id'], cat['name']) for cat in categories_data]}")
    
    # Determine the output cache file path based on dataset type
    if output_dataset_type == 'train':
        output_cache_file = TRAIN_CACHE_PATH
    elif output_dataset_type == 'val':
        output_cache_file = VAL_CACHE_PATH
    else: # 'test'
        output_cache_file = TEST_CACHE_PATH
    
    os.makedirs(os.path.dirname(output_cache_file), exist_ok=True)
    
    debug_output_directory = None
    if debug_mode:
        debug_output_directory = os.path.join(DEBUG_OUTPUT_DIR, "visor_preprocessing_debug", output_dataset_type)
        os.makedirs(debug_output_directory, exist_ok=True)
        print(f"Debug images will be saved to: {debug_output_directory}")
    
    processed_frames_list = []
    
    for img_info in tqdm(images_data, desc=f"Processing {output_dataset_type} images"):
        frame_data_dict = process_frame(img_info, annotations_data, categories_data, base_dataset_path, debug_output_directory)
        if frame_data_dict:
            processed_frames_list.append(frame_data_dict)
    
    if processed_frames_list:
        np.save(output_cache_file, processed_frames_list)
        print(f"Saved a total of {len(processed_frames_list)} processed frames to {output_cache_file}")
        return output_cache_file
    else:
        print(f"No valid frames processed for {output_dataset_type}. Output file {output_cache_file} not created or is empty.")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='VISOR Dataset Preprocessing Script')
    parser.add_argument('--dataset_type', choices=['train', 'val', 'test'], default='train',
                      help='Type of dataset to process (train, val, or test), corresponds to .json file name.')
    parser.add_argument('--debug', action='store_true', help='Save debug images during processing.')
    args = parser.parse_args()
    
    # Construct path to the JSON annotation file
    json_annotation_file = os.path.join(HOT3D_DATASET_PATH, "annotations", f"{args.dataset_type}.json")
    
    if not os.path.exists(json_annotation_file):
        print(f"Error: Annotation file not found at {json_annotation_file}")
        return

    process_visor_dataset(json_annotation_file, HOT3D_DATASET_PATH, args.dataset_type, args.debug)

if __name__ == "__main__":
    main()