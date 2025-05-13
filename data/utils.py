import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def get_clip_directories(dataset_path, max_clips=None):
    """
    Gets clip directories from the dataset path.
    
    Args:
        dataset_path: Path to the directory containing clips (e.g., train_aria).
        max_clips: Maximum number of clips to process (None or 'all' to process all).
    
    Returns:
        List of paths to clip directories.
    """
    # Find all directories in the dataset path that start with 'clip-'
    clip_dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) 
                if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('clip-')]
    
    clip_dirs.sort()
    
    if max_clips is not None and max_clips != 'all':
        try:
            num_to_take = int(max_clips)
            clip_dirs = clip_dirs[:num_to_take]
        except ValueError:
            print(f"Warning: max_clips value '{max_clips}' is not a valid number. Processing all clips.")

    
    print(f"Found {len(clip_dirs)} clips in {dataset_path}") # Kept as it's informative for a script run
    
    return clip_dirs

def decode_rle(rle_data, height, width):
    """
    Decodes a mask in RLE format.
    
    The RLE format in HOT3D (or similar datasets) consists of [index, count] pairs where:
    - index is the direct pixel index in the flattened image.
    - count is the number of consecutive pixels to activate.
    
    Args:
        rle_data: List of RLE [index, count] pairs.
        height: Height of the image.
        width: Width of the image.
    
    Returns:
        Binary mask (numpy array).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(0, len(rle_data), 2):
        pixel_idx = rle_data[i]
        count = rle_data[i+1]
        
        for j in range(count):
            curr_y = (pixel_idx + j) // width
            curr_x = (pixel_idx + j) % width
            
            # Boundary check for safety, though well-formed RLE shouldn't exceed bounds
            if 0 <= curr_y < height and 0 <= curr_x < width:
                mask[curr_y, curr_x] = 1
            # else:
            #     print(f"Warning: RLE decoding out of bounds: y={curr_y}, x={curr_x} for H={height}, W={width}")
    
    return mask

def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    Handles both [x1, y1, x2, y2] and [x, y, w, h] formats.
    
    Args:
        box1: Box [x1, y1, x2, y2] or [x, y, w, h].
        box2: Box [x1, y1, x2, y2] or [x, y, w, h].
    
    Returns:
        IoU value.
    """
    
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    if len(box1) == 4 and (box1[2] <= box1[0] or box1[3] <= box1[1]): # Likely [x,y,w,h] if w or h is not x2,y2
         b1_x2 = box1[0] + box1[2]
         b1_y2 = box1[1] + box1[3]

    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    if len(box2) == 4 and (box2[2] <= box2[0] or box2[3] <= box2[1]):
         b2_x2 = box2[0] + box2[2]
         b2_y2 = box2[1] + box2[3]

    # Calculate intersection coordinates
    x1_inter = max(b1_x1, b2_x1)
    y1_inter = max(b1_y1, b2_y1)
    x2_inter = min(b1_x2, b2_x2)
    y2_inter = min(b1_y2, b2_y2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter: # No overlap
        return 0.0
    
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def is_object_in_hand(obj_data, hands_data, camera_id):
    """
    Determines if an object is in hand based on HOT3D dataset criteria (or similar).
    
    Args:
        obj_data: Object data from objects.json.
        hands_data: Hands data from hands.json.
        camera_id: Camera ID string.
    
    Returns:
        True if the object is considered in hand, False otherwise.
    """
    # Check if any hand data is present for the frame
    if "left" not in hands_data and "right" not in hands_data:
        return False
    
    # 1. Check mask overlap (if available)
    if "masks_modal" in obj_data and camera_id in obj_data["masks_modal"]:
        obj_mask_info = obj_data["masks_modal"][camera_id]
        obj_mask = decode_rle(obj_mask_info["rle"], obj_mask_info["height"], obj_mask_info["width"])
        
        # Create a combined mask for both hands
        combined_hand_mask = None
        
        for hand_type in ["left", "right"]:
            if hand_type in hands_data and "masks" in hands_data[hand_type] and camera_id in hands_data[hand_type]["masks"]:
                hand_mask_info = hands_data[hand_type]["masks"][camera_id]
                current_hand_mask_decoded = decode_rle(hand_mask_info["rle"], hand_mask_info["height"], hand_mask_info["width"])
                
                if combined_hand_mask is None:
                    combined_hand_mask = current_hand_mask_decoded
                else:
                    # Union of masks from both hands
                    combined_hand_mask = np.logical_or(combined_hand_mask, current_hand_mask_decoded).astype(np.uint8)
        
        # If hand masks are available, check for overlap with object mask
        if combined_hand_mask is not None and combined_hand_mask.sum() > 0 and obj_mask.sum() > 0:
            intersection_pixels = np.logical_and(obj_mask, combined_hand_mask).sum()
            if intersection_pixels > 0:
                # If there's at least one pixel of overlap, consider the object in hand
                return True
    
    # 2. Check 3D distance (if available)
    if "T_world_from_object" in obj_data and obj_data["T_world_from_object"] is not None:
        obj_position_world = np.array(obj_data["T_world_from_object"]["translation_xyz"])
        
        min_hand_obj_distance = float('inf')
        for hand_type in ["left", "right"]:
            if hand_type in hands_data:
                hand_world_position = None
                
                # Try 'mano_pose' first (often more detailed)
                if "mano_pose" in hands_data[hand_type] and hands_data[hand_type]["mano_pose"] and \
                   "wrist_xform" in hands_data[hand_type]["mano_pose"] and \
                   hands_data[hand_type]["mano_pose"]["wrist_xform"] is not None:
                    # Assuming wrist_xform is a 4x4 matrix or list of 16 elements
                    wrist_transform_data = hands_data[hand_type]["mano_pose"]["wrist_xform"]
                    if isinstance(wrist_transform_data, list) and len(wrist_transform_data) == 16: # Flat list
                        hand_world_position = np.array(wrist_transform_data[12:15]) # Tx, Ty, Tz for column-major or row-major (check format)
                                                                                 # Assuming [R R R Tx; R R R Ty; R R R Tz; 0 0 0 1]
                                                                                 # Or if it's [Tx Ty Tz], then indices 3:6 is wrong.
                                                                                 # Let's assume it's the translation part of a 4x4 matrix
                                                                                 # If it's stored as [x,y,z, R, P, Y, scale...], then it's different
                                                                                 # The original `[3:6]` was likely for `[r,p,y,x,y,z]` kind of list.
                                                                                 # For a 4x4 matrix: wrist_xform_matrix = np.array(wrist_transform_data).reshape(4,4)
                                                                                 # hand_world_position = wrist_xform_matrix[:3, 3]
                        if len(wrist_transform_data) >= 6: hand_world_position = np.array(wrist_transform_data[3:6])


                # Try 'umetrack_pose' as a fallback
                elif "umetrack_pose" in hands_data[hand_type] and hands_data[hand_type]["umetrack_pose"] and \
                     "wrist_xform" in hands_data[hand_type]["umetrack_pose"] and \
                     hands_data[hand_type]["umetrack_pose"]["wrist_xform"] is not None:
                    wrist_xform_data = hands_data[hand_type]["umetrack_pose"]["wrist_xform"]
                    # Umetrack often provides a 4x4 matrix
                    if isinstance(wrist_xform_data, (list, np.ndarray)) and np.array(wrist_xform_data).size == 16:
                        wrist_xform_matrix = np.array(wrist_xform_data).reshape((4, 4))
                        hand_world_position = wrist_xform_matrix[:3, 3] # Translation part
                
                if hand_world_position is not None:
                    distance = np.linalg.norm(obj_position_world - hand_world_position)
                    min_hand_obj_distance = min(min_hand_obj_distance, distance)
        
        # Consider in hand if the 3D distance is less than, e.g., 1 cm (0.01 m)
        # This threshold might need tuning based on dataset specifics.
        if min_hand_obj_distance < DISTANCE_THRESHOLD: # Using DISTANCE_THRESHOLD from config
            return True
    
    # 3. Fallback method based on IoU between amodal bounding boxes
    if "boxes_amodal" in obj_data and camera_id in obj_data["boxes_amodal"]:
        obj_box_amodal = obj_data["boxes_amodal"][camera_id] # Expected [x, y, w, h]
        
        for hand_type in ["left", "right"]:
            if hand_type in hands_data and "boxes_amodal" in hands_data[hand_type] and \
               camera_id in hands_data[hand_type]["boxes_amodal"]:
                hand_box_amodal = hands_data[hand_type]["boxes_amodal"][camera_id] # Expected [x, y, w, h]
                
                # calculate_iou expects [x1,y1,x2,y2] or [x,y,w,h] and handles conversion
                iou = calculate_iou(obj_box_amodal, hand_box_amodal)
                if iou > IOU_THRESHOLD: # Using IOU_THRESHOLD from config
                    return True
    
    return False

def save_debug_image(image_rgb, mask_binary, output_path=None):
    """
    Saves an image with an overlaid mask for debugging purposes.
    
    Args:
        image_rgb: Original image (RGB format, HxWxC numpy array).
        mask_binary: Binary mask (HxW numpy array, 0s and 1s).
        output_path (str, optional): Path to save the debug image. If None, does nothing.
    """
    if output_path is None:
        return
    
    vis_image = image_rgb.copy()
    
    # Ensure the image is 3-channel RGB for color overlay
    if len(vis_image.shape) == 2: # Grayscale
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
    elif vis_image.shape[2] == 4: # RGBA
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGBA2RGB)

    mask_color_rgb = np.array(BLUE_MASK_COLOR) # Assuming BLUE_MASK_COLOR is RGB
    alpha_blend = MASK_ALPHA # Opacity from config
    
    # Apply the mask overlay
    mask_boolean = mask_binary > 0 # Ensure mask is boolean
    if np.any(mask_boolean):
        # Create an image with only the colored mask
        colored_mask_layer = np.zeros_like(vis_image)
        colored_mask_layer[mask_boolean] = mask_color_rgb
        
        # Blend with transparency
        vis_image[mask_boolean] = ((1 - alpha_blend) * vis_image[mask_boolean] + 
                                   alpha_blend * colored_mask_layer[mask_boolean]).astype(np.uint8)
    
    # Save the image (OpenCV expects BGR)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))