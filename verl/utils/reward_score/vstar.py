import json
import numpy as np
from tqdm import tqdm
import math
import ast
import re
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# --- Parsing Functions for Model Output ---

def parse_predicted_timestamps(text: str) -> Optional[List[float]]:
    """Extract the last list/tuple of two numbers from text."""
    if not text:
        return None

    # Try finding patterns like [10.0, 17.9] or (10.0, 17.9)
    # Regex to find list/tuple-like structures with two numbers
    matches = re.findall(r'[\(\[]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\)\]]', text)

    if not matches:
        # Fallback: Try ast.literal_eval on the entire string or parts if regex fails
        try:
            potential_list = ast.literal_eval(text)
            if isinstance(potential_list, (list, tuple)) and len(potential_list) == 2 and all(isinstance(x, (int, float)) for x in potential_list):
                return sorted([float(potential_list[0]), float(potential_list[1])])
        except (ValueError, SyntaxError):
            pass # Ignore if eval fails
        return None

    # Return the last valid match found by regex
    try:
        last_match = matches[-1]
        return sorted([float(last_match[0]), float(last_match[1])])
    except (ValueError, IndexError):
        return None

def parse_predicted_bboxes(text: str) -> Optional[Dict[str, List[int]]]:
    """Extract bounding boxes from text, expecting a dict format like {'frame_id': [xmin, ymin, xmax, ymax], ...}."""
    if not text:
        return None

    # Try to find dictionary-like structures using regex
    # This regex is basic and might need refinement depending on actual model output format
    dict_match = re.search(r'\{[^{}]*\}', text)

    if dict_match:
        dict_str = dict_match.group(0)
        try:
            parsed_dict = ast.literal_eval(dict_str)
            if isinstance(parsed_dict, dict):
                # Validate structure: keys should be frame_ids (strings/ints), values should be lists of 4 numbers
                valid_bboxes = {}
                for k, v in parsed_dict.items():
                    frame_id = str(k) # Ensure frame_id is string
                    if isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
                        # Convert coords to int, ensure non-negative
                        coords = [max(0, int(c)) for c in v]
                        # Basic validity check (xmax > xmin, ymax > ymin)
                        if coords[2] > coords[0] and coords[3] > coords[1]:
                            valid_bboxes[frame_id] = coords
                        else:
                            logger.warning(f"Parsed invalid bbox coords for frame {frame_id}: {coords}")
                    else:
                        logger.warning(f"Skipping invalid bbox value for frame {frame_id}: {v}")

                if valid_bboxes:
                    return valid_bboxes
                else:
                    logger.warning(f"Parsed dict, but no valid bboxes found: {dict_str}")
                    return None
            else:
                logger.warning(f"ast.literal_eval did not return a dict: {dict_str}")

        except (ValueError, SyntaxError, TypeError) as e:
            logger.warning(f"Failed to parse predicted bbox dict string using ast.literal_eval: '{dict_str}'. Error: {e}")
            pass # Fall through if eval fails

    # Fallback or if no dict found
    logger.warning(f"Could not parse bounding box dictionary from text: {text}")
    return None

# --- Existing IoU Calculation Functions (keep as is) ---

def calculate_temporal_iou(gt_range, pred_range):
    """compute Temporal IoU"""
    if not pred_range:  # If pred_range is None or empty
        return 0.0  # Return default 0.0
    
    # If pred_range is a string, then try to convert to a list
    if isinstance(pred_range, str):
        try:
            pred_range = ast.literal_eval(pred_range)
        except (ValueError, SyntaxError):
            logger.warning(f"Temporal IoU: Failed to parse pred_range string: {pred_range}")
            return 0.0  #  The conversion fails and returns the default value of 0.0
    
    # Ensure that pred_range is a list or tuple of two values
    if not isinstance(pred_range, (list, tuple)) or len(pred_range) != 2 or \
    not all(isinstance(x, (int, float)) for x in pred_range):
        logger.warning(f"Temporal IoU: Invalid pred_range format: {pred_range}")
        return 0.0  # is not a valid values, returns the default value of 0.0

    gt_start, gt_end = gt_range
    pred_start, pred_end = pred_range
    if pred_start > pred_end:
        return 0.0
    intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = max(gt_end, pred_end) - min(gt_start, pred_start)
    # Add epsilon to union to prevent division by zero if gt_range and pred_range are identical points
    print(f"Intersection: {intersection}, Union: {union} for gt_range: {gt_range} and pred_range: {pred_range}")
    return intersection / (union + 1e-6) if union >= 0 else 0.0

def compute_iou(gt_bbox, pred_bbox):
    """Calculate the IoU for two boxes"""
    if not isinstance(pred_bbox, (list, tuple)) or len(pred_bbox) != 4:
        return 0.0
    
    # get GT bbox coordinates
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox['xmin'], gt_bbox['ymin'], gt_bbox['xmax'], gt_bbox['ymax']
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_bbox

    # calculate intersection
    x1 = max(gt_xmin, pred_xmin)
    y1 = max(gt_ymin, pred_ymin)
    x2 = min(gt_xmax, pred_xmax)
    y2 = min(gt_ymax, pred_ymax)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # calculate union
    gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    union = gt_area + pred_area - intersection

    return intersection / union if union > 0 else 0.0

def calculate_bbox_iou(gt_bbox, pred_bboxes):
    """Calculate single BBox IoU, support multiple prediction boxes to take max IoU"""
    try:
        if not pred_bboxes:
            return 0.0
        
        # Handling of individual boxes
        if isinstance(pred_bboxes[0], (int, float)) and len(pred_bboxes) == 4:
            pred_bboxes = [pred_bboxes]
        
        # Calculate the IoU for all prediction frames and return the maximum value
        return max([compute_iou(gt_bbox, pred_bbox) for pred_bbox in pred_bboxes])
    except:
        return 0.0

def calculate_spatial_metrics(gt_bboxes, pred_bboxes):
    """Compute vIoU and AP"""
    if not pred_bboxes:  # Checks if pred_bboxes are None or empty.
        return [0.0] * 5, 0.0  # Return default: 0 for all APs, 0 for m_vIoU

    iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    ious = []
    aps = []
    for gt_bbox_entry in gt_bboxes:
        for frame_id, gt_bbox in gt_bbox_entry.items():
            frame_id = frame_id.split("_")[0]
            if frame_id in pred_bboxes:
                pred_bbox = pred_bboxes[frame_id]
                iou = calculate_bbox_iou(gt_bbox, pred_bbox)
                ious.append(iou)
            else:
                ious.append(0.0)
    mIoU = np.mean(ious) if ious else 0.0

    for threshold in iou_thresholds:
        scores = [1 if iou >= threshold else 0 for iou in ious]
        if len(ious) > 0:
            aps.append(np.mean(scores))
        else:
            aps.append(0.0)
    return aps, mIoU

# --- Main Scoring Function ---

def compute_score(solution_str: str, ground_truth_str: str) -> float:
    """Compute score based on task type (temporal or spatial IoU)."""
    print(f"Computing score for solution: {solution_str} and ground_truth: {ground_truth_str}")
    if not solution_str or not ground_truth_str:
        return 0.0

    try:
        # Check if ground_truth is a JSON with style information
        try:
            # Try parsing as a JSON object first to see if we can extract a style
            ground_truth_obj = json.loads(ground_truth_str)
            if isinstance(ground_truth_obj, dict) and "style" in ground_truth_obj:
                style = ground_truth_obj.get("style")
                actual_ground_truth = ground_truth_obj.get("value", ground_truth_str)
                
                # Use style to route to specific scoring function if available
                if style == "vstar_temporal_iou":
                    predicted_value = parse_predicted_timestamps(solution_str)
                    if predicted_value is None:
                        logger.warning(f"Could not parse temporal prediction: {solution_str}")
                        return 0.0
                    temporal_output = calculate_temporal_iou(actual_ground_truth, predicted_value)
                    print(f"Temporal output: {temporal_output}")
                    return temporal_output
                    
                elif style == "vstar_spatial_iou":
                    predicted_value = parse_predicted_bboxes(solution_str)
                    if predicted_value is None:
                        logger.warning(f"Could not parse spatial prediction: {solution_str}")
                        return 0.0
                    # Fix: actual_ground_truth should be correctly formatted for calculate_spatial_metrics
                    # The function expects a list of dict entries (from the original format),
                    # not just a dict mapping frame_id to bbox coordinates
                    formatted_gt = []
                    for frame_id, coords in actual_ground_truth.items():
                        bbox_dict = {"xmin": coords[0], "ymin": coords[1], "xmax": coords[2], "ymax": coords[3]}
                        formatted_gt.append({frame_id: bbox_dict})
                    # Use the first AP threshold (0.1) score for consistency with legacy path
                    spatial_output = calculate_spatial_metrics(formatted_gt, predicted_value)[0][0]
                    print(f"Spatial output: {spatial_output}")
                    return spatial_output
        except (json.JSONDecodeError, TypeError):
            # Not a JSON with style, proceed with regular processing
            pass

    except json.JSONDecodeError:
        logger.error(f"Failed to parse ground_truth JSON: {ground_truth_str}")
        return 0.0
    except Exception as e:
        logger.error(f"Error during score computation: {e}", exc_info=True)
        return 0.0