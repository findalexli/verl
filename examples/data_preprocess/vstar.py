#!/usr/bin/env python3
# Copyright 2024 ByteDance and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preprocess V-STAR dataset annotations.

This script processes annotations from the V-STAR dataset, extracts frames from the videos
at a specified FPS, and saves the frames as image files. The resulting dataset stores lists
of frame paths instead of video paths, which significantly speeds up data loading during training.
"""

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
from collections import Counter
import math

import datasets
from tqdm import tqdm
import logging
import numpy as np
import torch
from verl.utils.dataset.vision_utils import process_video
from qwen_vl_utils import fetch_video
from PIL import Image
from io import BytesIO
import torchvision.transforms.functional as TF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Video Utility Functions ---

def find_ffprobe_path():
    """Find the ffprobe executable path"""
    # Try common locations
    common_locations = [
        "ffprobe",  # Default PATH
        "/usr/bin/ffprobe",
        "/usr/local/bin/ffprobe",
        "/opt/homebrew/bin/ffprobe",  # macOS Homebrew
    ]
    
    for location in common_locations:
        try:
            # Check if executable works
            subprocess.check_output([location, "-version"], stderr=subprocess.STDOUT)
            logger.info(f"Found ffprobe at: {location}")
            return location
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    
    logger.warning("ffprobe not found in common locations. Video metadata extraction will be limited.")
    return None

# Initialize ffprobe path
FFPROBE_PATH = find_ffprobe_path()

def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe"""
    try:
        if video_path.startswith("file://"):
            video_path = video_path[7:]
            
        if not FFPROBE_PATH:
            logger.warning(f"ffprobe not available. Using fallback method for {video_path}")
            # Fallback to a simple file existence check
            if not os.path.exists(video_path):
                return 0
            # Return a default duration since we can't extract it
            return 30.0  # Assume 30 seconds as fallback
            
        cmd = [FFPROBE_PATH, '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', video_path]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(output)
        return float(data['format']['duration'])
    except Exception as e:
        logger.error(f"Could not get video duration for {video_path}: {e}")
        # If file exists but probe failed, return a default value
        if os.path.exists(video_path):
            return 30.0
        return 0

def get_video_dimensions(video_path):
    """Get video width and height using ffprobe"""
    try:
        if video_path.startswith("file://"):
            video_path = video_path[7:]
            
        if not FFPROBE_PATH:
            logger.warning(f"ffprobe not available. Using fallback dimensions for {video_path}")
            # Return default dimensions if ffprobe isn't available
            return 1280, 720  # Return HD resolution as fallback
            
        cmd = [FFPROBE_PATH, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'json', video_path]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(output)
        if 'streams' in data and len(data['streams']) > 0:
            width = int(data['streams'][0]['width'])
            height = int(data['streams'][0]['height'])
            return width, height
        logger.warning(f"Could not parse dimensions from ffprobe output for {video_path}")
        return 1280, 720  # Return HD resolution as fallback
    except Exception as e:
        logger.error(f"Could not get video dimensions for {video_path}: {e}")
        return 1280, 720  # Return HD resolution as fallback

def is_valid_video(video_path):
    """Check if a video file is valid using ffprobe without processing the content"""
    try:
        if video_path.startswith("file://"):
            video_path = video_path[7:]
            
        # If ffprobe isn't available, just check if the file exists
        if not FFPROBE_PATH:
            exists = os.path.exists(video_path) and os.path.getsize(video_path) > 0
            logger.info(f"ffprobe not available. File existence check for {video_path}: {'Passed' if exists else 'Failed'}")
            return exists
            
        # Simple check to validate video streams exist and can be read
        cmd = [FFPROBE_PATH, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_type', '-of', 'json', video_path]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(output)
        # Check if we have at least one video stream
        return 'streams' in data and len(data['streams']) > 0 and data['streams'][0].get('codec_type') == 'video'
    except Exception as e:
        logger.warning(f"Invalid video file {video_path}: {e}")
        # If ffprobe fails but file exists, consider it valid and let processing handle it
        if FFPROBE_PATH is None and os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            return True
        return False

def calculate_video_tokens(video_path, fps=1):
    """Calculate total tokens for a video based on dimensions and duration"""
    duration = get_video_duration(video_path)
    width, height = get_video_dimensions(video_path)

    if duration <= 0 or width <= 0 or height <= 0:
        logger.warning(f"Skipping token calculation due to invalid metadata for {video_path} (duration={duration}, w={width}, h={height})")
        return None

    num_frames = int(duration * fps)
    # Ensure width and height are divisible by 28 for token calculation
    tokens_per_frame = (width // 28) * (height // 28)
    total_tokens = tokens_per_frame * num_frames
    logger.info(f"Video: {os.path.basename(video_path)}, duration: {duration:.2f}s, dims: {width}x{height}, fps: {fps}, num_frames: {num_frames}, tokens/frame: {tokens_per_frame}, total_tokens: {total_tokens}")
    return {
        'duration': duration,
        'width': width,
        'height': height,
        'num_frames': num_frames,
        'tokens_per_frame': tokens_per_frame,
        'total_tokens': total_tokens
    }

def extract_and_save_frames(
    vid: str,
    video_path: str,
    frames_base_dir: Path,
    fps: float,
    total_pixels: int,
    min_pixels: int
) -> Optional[List[str]]:
    """Reads video, extracts frames using fetch_video, saves them, and returns paths."""
    try:
        # 1. Fetch the frame tensor using qwen_vl_utils
        # We construct the dictionary expected by fetch_video
        video_info_dict = {
            "type": "video",
            "video": f"file://{video_path}", # Use file URI
            "fps": fps,
            "total_pixels": total_pixels,
            "min_pixels": min_pixels
        }
        # fetch_video returns a tensor [n_frames, C, H, W] or list of PIL Images
        # Let's ensure it returns a tensor for consistent processing
        # Note: fetch_video already handles sampling and resizing based on parameters
        frame_data = fetch_video(video_info_dict)

        if not isinstance(frame_data, torch.Tensor) or frame_data.ndim != 4:
            logger.error(f"fetch_video did not return expected tensor for {vid}. Got {type(frame_data)}")
            return None

        num_frames = frame_data.shape[0]
        if num_frames == 0:
            logger.warning(f"fetch_video returned 0 frames for {vid}. Skipping.")
            return None

        # 2. Create output directory for this video's frames
        video_frames_dir = frames_base_dir / vid
        video_frames_dir.mkdir(parents=True, exist_ok=True)

        # 3. Save frames and collect paths
        frame_paths = []
        for i in range(num_frames):
            frame_tensor = frame_data[i] # Shape [C, H, W]
            # Convert tensor to PIL Image (ensure it's in range 0-255 if needed)
            # fetch_video should return float tensor in [0,1], convert to uint8
            pil_image = TF.to_pil_image((frame_tensor * 255).byte())
            frame_filename = f"frame_{i:06d}.jpg" # Use 6 digits for padding
            frame_output_path = video_frames_dir / frame_filename
            pil_image.save(frame_output_path, quality=95)
            frame_paths.append(f"file://{frame_output_path.absolute()}")

        logger.info(f"Saved {num_frames} frames for {vid} to {video_frames_dir}")
        return frame_paths

    except Exception as e:
        logger.error(f"Error extracting/saving frames for {vid} from {video_path}: {e}", exc_info=True)
        return None

# --- V-STAR Specific Processing ---

def format_ground_truth(task_type: str, data: Dict) -> Optional[str]:
    """Format ground truth based on task type."""
    if task_type == "temporal":
        gt = data.get("timestamps")
        if gt and isinstance(gt, list) and len(gt) == 2 and all(isinstance(x, (int, float)) for x in gt):
             # Ensure order [min, max]
             gt = sorted(gt)
             return json.dumps({"style": "vstar_temporal_iou", "value": gt})
        else:
             logger.warning(f"Invalid or missing 'timestamps' for temporal task in vid {data.get('vid')}: {gt}")
             return None
    elif task_type == "spatial":
        gt = data.get("bboxes")
        # Expecting list of dicts like [{"frame_id_str": {"xmin":..., "ymin":..., "xmax":..., "ymax":...}}]
        if gt and isinstance(gt, list) and len(gt) > 0 and isinstance(gt[0], dict):
            # Basic validation of structure
            try:
                # Check first bbox entry for validity
                first_entry = gt[0]
                frame_key = list(first_entry.keys())[0]
                bbox_dict = first_entry[frame_key]
                coords = [bbox_dict['xmin'], bbox_dict['ymin'], bbox_dict['xmax'], bbox_dict['ymax']]
                if not all(isinstance(c, int) for c in coords):
                     raise ValueError("Coordinates are not integers")
                # Convert GT bboxes to the format needed by evaluation (dict mapping frame_id to list)
                # Also ensure coordinates are within valid range [0, 1] if normalized, or use pixel values directly
                # The provided eval script seems to use pixel values. Let's assume pixel values for now.
                formatted_bboxes = {}
                for entry in gt:
                     frame_key = list(entry.keys())[0]
                     frame_id = frame_key.split('_')[0] # Extract frame index part
                     bbox_dict = entry[frame_key]
                     # Ensure coordinates are valid pixel values (non-negative)
                     coords = [
                         max(0, bbox_dict['xmin']),
                         max(0, bbox_dict['ymin']),
                         max(0, bbox_dict['xmax']),
                         max(0, bbox_dict['ymax'])
                     ]
                     # Check if coords are valid (xmax > xmin, ymax > ymin) - skip if not
                     if coords[2] <= coords[0] or coords[3] <= coords[1]:
                          logger.warning(f"Invalid GT bbox coords in vid {data.get('vid')}, frame {frame_key}: {coords}. Skipping.")
                          continue

                     # Store as list [xmin, ymin, xmax, ymax]
                     if frame_id not in formatted_bboxes:
                          formatted_bboxes[frame_id] = []
                     # The eval function `calculate_bbox_iou` handles multiple *predicted* boxes per frame.
                     # Let's keep the structure simple: one box per frame_id based on GT.
                     # If multiple entries exist for the same frame_id, the last one will overwrite.
                     formatted_bboxes[frame_id] = coords # Store the single GT box

                if not formatted_bboxes:
                     logger.warning(f"No valid GT bboxes found after formatting for vid {data.get('vid')}")
                     return None

                return json.dumps({"style": "vstar_spatial_iou", "value": formatted_bboxes})

            except (KeyError, IndexError, TypeError, ValueError) as e:
                 logger.warning(f"Invalid 'bboxes' structure for spatial task in vid {data.get('vid')}: {gt}. Error: {e}")
                 return None
        else:
             logger.warning(f"Invalid or missing 'bboxes' for spatial task in vid {data.get('vid')}: {gt}")
             return None
    else:
        return None

def process_vstar_annotation(
    annotation: Dict,
    video_base_dir: Path,
    data_source_name: str,
    calculate_tokens: bool,
    max_video_tokens: int,
    total_pixels: int,
    min_pixels: int,
    fps: float,
    frames_base_dir: Path
) -> List[Dict]:
    """Processes a single V-STAR annotation entry."""
    examples = []
    vid = annotation.get("vid")
    if not vid:
        logger.warning("Skipping annotation due to missing 'vid'")
        return []

    # Construct video path - assumes videos are named {vid}.mp4
    video_path = video_base_dir / f"{vid}.mp4"
    if not video_path.exists():
        logger.warning(f"Video file not found for vid {vid} at {video_path}, skipping.")
        return []

    video_path_str = str(video_path.absolute())

    # --- Validate video file using ffprobe ---
    if not is_valid_video(video_path_str):
        logger.warning(f"Invalid or corrupted video file for vid {vid} at {video_path}, skipping.")
        return []
    # --- End video validation ---

    # --- Extract and Save Frames ---
    frame_paths = extract_and_save_frames(
        vid,
        video_path_str,
        frames_base_dir,
        fps, # fps from args
        total_pixels,
        min_pixels
    )
    if not frame_paths:
        logger.warning(f"Could not extract frames for {vid}. Skipping annotation.")
        return []
    # ---

    # Calculate video tokens if requested (for filtering/statistics)
    # Note: Token calculation might need adjustment if based on original video
    video_tokens_info = None
    if calculate_tokens:
        video_tokens_info = calculate_video_tokens(video_path_str, fps=fps)
        if video_tokens_info and video_tokens_info["total_tokens"] > max_video_tokens:
            logger.info(f"Skipping {video_path_str} - original video token count {video_tokens_info['total_tokens']} exceeds limit of {max_video_tokens}")
            # Decide whether to skip based on original video size or frame count?
            # Let's keep skipping based on original for now.
            return [] # Skip this annotation entirely if video too long based on tokens

    # Get video duration
    video_duration = video_tokens_info["duration"] if video_tokens_info else get_video_duration(video_path_str)
    video_duration_rounded = round(video_duration, 2)

    # --- Create Temporal Example ---
    temporal_question = annotation.get("temporal_question")
    ground_truth_temporal_str = format_ground_truth("temporal", annotation)

    if temporal_question and ground_truth_temporal_str:
        # Use the author's temporal prompt template with explicit output format requirements
        temporal_prompt = f"""This video is {video_duration_rounded} seconds long. Answer the question about the video: {temporal_question}
Output the start and end moment timestamps in the format [start_time, end_time] where both values are in seconds.
Your response must contain exactly one pair of timestamps in square brackets or parentheses like [10.5, 15.2] or (10.5, 15.2).
"""
        
        temporal_example = {
            "data_source": data_source_name,
            "prompt": [{"role": "user", "content": f"<video>{temporal_prompt}"}],
            "videos": [{
                "type": "video",
                "video": frame_paths, # Use list of frame paths
                "fps": fps
                # Remove pixel constraints here, handled by frame extraction/fetch_image
            }],
            "ability": "vstar_temporal",
            "reward_model": {"style": "rule", "ground_truth": ground_truth_temporal_str},
            "extra_info": {
                "capability": "vstar_temporal",
                "vid": vid,
                "domain": annotation.get("domain", "unknown"),
                "original_question": annotation.get("question", ""),
                "task_type": "temporal",
                "original_video_file": video_path_str, # Keep original path
                "frames_dir": str(frames_base_dir / vid) # Add frames directory
            },
        }
        if video_tokens_info:
            temporal_example["extra_info"]["video_tokens"] = video_tokens_info
        examples.append(temporal_example)

    # --- Create Spatial Example ---
    spatial_question = annotation.get("spatial_question") # Use the first spatial question
    ground_truth_spatial_str = format_ground_truth("spatial", annotation)

    if spatial_question and ground_truth_spatial_str:
        # Get timestamps for time range if available
        timestamps = annotation.get("timestamps", [0, int(video_duration)])
        st, et = math.ceil(timestamps[0]), math.floor(timestamps[1])
        time_range = list(range(st, et + 1))
        
        # Get video dimensions
        w, h = get_video_dimensions(video_path_str)
        if video_tokens_info:
            w, h = video_tokens_info["width"], video_tokens_info["height"]
        
        # Use the author's spatial prompt template with explicit output format requirements
        spatial_prompt = f"""Please answer the question about the video: {spatial_question} with a series of bounding boxes in [x1, y1, x2, y2] format.
For each whole second within the time range {time_range} provided (inclusive of the boundaries), output a series of bounding boxes of the object in JSON format.
The keys should be the whole seconds (as strings), and the values should be the box in [x1, y1, x2, y2] format.
Your response must contain exactly one JSON dictionary with no other text.
Example output: {{"{time_range[0]}": [x1, y1, x2, y2], "{time_range[0]+1}": [x1, y1, x2, y2], ...}}"""
        
        spatial_example = {
            "data_source": data_source_name,
            "prompt": [{"role": "user", "content": f"<video>{spatial_prompt}"}],
            "videos": [{
                "type": "video",
                "video": frame_paths, # Use list of frame paths
                "fps": fps
                # Remove pixel constraints here, handled by frame extraction/fetch_image
            }],
            "ability": "vstar_spatial",
            "reward_model": {"style": "rule", "ground_truth": ground_truth_spatial_str},
            "extra_info": {
                "capability": "vstar_spatial",
                "vid": vid,
                "domain": annotation.get("domain", "unknown"),
                "original_question": annotation.get("question", ""),
                "task_type": "spatial",
                "original_video_file": video_path_str, # Keep original path
                "frames_dir": str(frames_base_dir / vid), # Add frames directory
                "time_range": time_range,
            },
        }
        if video_tokens_info:
            spatial_example["extra_info"]["video_tokens"] = video_tokens_info
        examples.append(spatial_example)

    if not examples:
         logger.warning(f"No valid temporal or spatial examples generated for vid {vid}")

    return examples


def main():
    parser = argparse.ArgumentParser(description="Preprocess V-STAR dataset annotations.")
    parser.add_argument(
        "--annotation_file",
        default="/home/alexli/data/V-STaR/V-STaR/data/V_STaR_test_release.jsonl",
        help="Path to the V-STAR JSON annotation file (one JSON object per line).",
    )
    parser.add_argument(
        "--video_base_dir",
        default="/home/alexli/data/V-STaR/videos",
        help="Path to the directory containing the V-STAR videos (e.g., named vid.mp4).",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/alexli/data/V-STaR/finetune_data",
        help="Directory to save the output parquet files.",
    )
    parser.add_argument(
        "--train_filename", default="vstar_train", help="Base name for the training output file (without extension)."
    )
    parser.add_argument(
        "--val_filename", default="vstar_val", help="Base name for the validation output file (without extension)."
    )
    parser.add_argument(
        "--val_percent", type=float, default=0.001, help="Percentage for validation set (0 to 1)."
    )
    parser.add_argument(
        "--data_source_name", default="vstar", help="Identifier for this data source."
    )
    parser.add_argument(
        "--num_workers", type=int, default=os.cpu_count(), help="Number of parallel workers."
    )
    parser.add_argument(
        "--max_annotations", type=int, default=None, help="Max annotations to process (for testing)."
    )
    parser.add_argument(
        "--calculate_tokens", action="store_true", help="Calculate video token stats."
    )
    parser.add_argument(
        "--token_stats_file", default="vstar_video_token_stats.json", help="File for token stats."
    )
    parser.add_argument(
        "--max_video_tokens", type=int, default=30000, help="Max video tokens allowed for filtering."
    )
    # Video processing parameters
    parser.add_argument(
        "--total_pixels", type=int, default=20480 * 28 * 28, 
        help="Maximum total pixels across all frames (default: 20480 * 28 * 28)."
    )
    parser.add_argument(
        "--min_pixels", type=int, default=16 * 28 * 28, 
        help="Minimum pixels per frame (default: 16 * 28 * 28)."
    )
    parser.add_argument("--fps", type=float, default=1.0, help="FPS for video processing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    # New arguments for slicing
    parser.add_argument(
        "--slice_type", 
        choices=["all", "temporal", "spatial"], 
        default="all",
        help="Type of examples to include (all, temporal, or spatial)."
    )
    parser.add_argument(
        "--frames_output_dir",
        default=None,
        help="Directory to save extracted frames. Defaults to a 'frames' subdirectory in output_dir.",
    )

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine frames output directory
    if args.frames_output_dir:
        frames_base_dir = Path(args.frames_output_dir)
    else:
        frames_base_dir = output_dir / "frames"
    frames_base_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving extracted frames to: {frames_base_dir}")

    # Add suffix based on slice_type if not "all"
    file_suffix = ""
    if args.slice_type != "all":
        file_suffix = f"_{args.slice_type}"
    
    train_parquet_path = output_dir / f"{args.train_filename}{file_suffix}.parquet"
    val_parquet_path = output_dir / f"{args.val_filename}{file_suffix}.parquet"

    # Load annotations
    video_base_dir = Path(args.video_base_dir)
    annotations = []
    try:
        with open(args.annotation_file, "r") as f:
            for i, line in enumerate(f):
                if args.max_annotations and i >= args.max_annotations:
                     logger.info(f"Reached max_annotations limit ({args.max_annotations}).")
                     break
                try:
                    annotations.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {i+1}: {e}")
    except FileNotFoundError:
        logger.error(f"Annotation file not found: {args.annotation_file}")
        return
    except Exception as e:
         logger.error(f"Error reading annotation file {args.annotation_file}: {e}")
         return

    if not annotations:
         logger.error("No valid annotations loaded. Exiting.")
         return

    logger.info(f"Loaded {len(annotations)} annotations.")

    # --- Process annotations in parallel
    all_examples = []
    
    # Add counters and timing for frame extraction metrics
    extraction_stats = {
        "total_videos": 0,
        "videos_processed": 0,
        "frames_extracted": 0,
        "failed_extractions": 0
    }
    
    logger.info(f"Starting to process {len(annotations)} annotations using {args.num_workers} workers")
    logger.info(f"Frame extraction mode enabled! Frames will be saved to {frames_base_dir}")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_vstar_annotation,
                anno,
                video_base_dir,
                args.data_source_name,
                args.calculate_tokens,
                args.max_video_tokens,
                args.total_pixels,
                args.min_pixels,
                args.fps,
                frames_base_dir
            )
            for anno in annotations
        ]
        for future in tqdm(futures, desc="Processing annotations", total=len(annotations)):
            try:
                examples = future.result()
                
                # Update extraction stats if examples were generated
                if examples:
                    extraction_stats["videos_processed"] += 1
                    # Count frames from the first example (both examples should use the same frames)
                    if len(examples) > 0 and "videos" in examples[0] and len(examples[0]["videos"]) > 0:
                        frames_count = len(examples[0]["videos"][0]["video"])
                        extraction_stats["frames_extracted"] += frames_count
                else:
                    extraction_stats["failed_extractions"] += 1
                
                extraction_stats["total_videos"] += 1
                all_examples.extend(examples)
            except Exception as e:
                # This might catch errors within the process_vstar_annotation call itself
                logger.error(f"Error processing an annotation: {e}", exc_info=True)
                extraction_stats["failed_extractions"] += 1
                extraction_stats["total_videos"] += 1

    # Log frame extraction statistics
    logger.info("Frame extraction complete!")
    logger.info(f"Videos processed: {extraction_stats['videos_processed']} / {extraction_stats['total_videos']} " +
                f"({extraction_stats['videos_processed']/max(1, extraction_stats['total_videos'])*100:.1f}%)")
    logger.info(f"Total frames extracted: {extraction_stats['frames_extracted']}")
    logger.info(f"Failed extractions: {extraction_stats['failed_extractions']}")
    logger.info(f"Average frames per video: {extraction_stats['frames_extracted']/max(1, extraction_stats['videos_processed']):.1f}")

    # Filter examples based on slice_type
    if args.slice_type == "temporal":
        all_examples = [ex for ex in all_examples if ex["ability"] == "vstar_temporal"]
        logger.info(f"Filtered to {len(all_examples)} temporal examples")
    elif args.slice_type == "spatial":
        all_examples = [ex for ex in all_examples if ex["ability"] == "vstar_spatial"]
        logger.info(f"Filtered to {len(all_examples)} spatial examples")
    
    logger.info(f"Total processed examples after filtering: {len(all_examples)}")

    if not all_examples:
        logger.warning("No valid examples were generated after filtering. Exiting.")
        return

    # --- Calculate token distribution (if needed) ---
    if args.calculate_tokens:
        logger.info("Calculating video token distribution...")
        # Gather token statistics from examples
        valid_token_infos = [ex["extra_info"]["video_tokens"] for ex in all_examples if ex["extra_info"].get("video_tokens")]
        if valid_token_infos:
             token_counts = [info["total_tokens"] for info in valid_token_infos]
             stats = {
                 "min": min(token_counts), 
                 "max": max(token_counts),
                 "mean": float(np.mean(token_counts)), 
                 "median": float(np.median(token_counts)),
                 "p95": float(np.percentile(token_counts, 95)), 
                 "p99": float(np.percentile(token_counts, 99)),
                 "distribution": dict(Counter([t // 1000 * 1000 for t in token_counts])) # Group by 1k
             }
             stats_file = output_dir / args.token_stats_file
             try:
                 with open(stats_file, 'w') as f:
                     json.dump({"token_stats": stats}, f, indent=2)
                 logger.info(f"Token statistics saved to {stats_file}")
                 logger.info(f"Token stats summary: Min={stats['min']}, Max={stats['max']}, Mean={stats['mean']:.2f}, Median={stats['median']}, P95={stats['p95']}, P99={stats['p99']}")
             except Exception as e:
                 logger.error(f"Failed to save token stats: {e}")
        else:
             logger.warning("No valid video token information found to calculate stats.")

    # Update temporal prompt format to be more explicit
    for example in all_examples:
        if example["ability"] == "vstar_temporal":
            # Find the user message in the prompt
            for message in example["prompt"]:
                if message["role"] == "user":
                    # Extract the existing prompt content
                    content = message["content"]
                    # Check if content has <video> tag
                    if "<video>" in content:
                        video_question = content.replace("<video>", "").strip()
                        # Replace with more explicit instructions
                        video_duration_rounded = "unknown"
                        if example["extra_info"].get("video_tokens", {}).get("duration"):
                            video_duration_rounded = round(example["extra_info"]["video_tokens"]["duration"], 2)
                        
                        # Extract the question text
                        question_text = ""
                        if "Answer the question about the video:" in video_question:
                            question_text = video_question.split("Answer the question about the video:")[1].strip()
                        
                        new_prompt = f"""<video>This video is {video_duration_rounded} seconds long. Answer the question about the video: {question_text}
IMPORTANT: Your ENTIRE response must be ONLY a pair of timestamps in the format [start_time, end_time] where both values are in seconds.
Example: [10.5, 15.2]
Do not include any other text, explanations, or code blocks in your response."""
                        
                        message["content"] = new_prompt
                        break
        
        elif example["ability"] == "vstar_spatial" and args.slice_type != "temporal":
            # Find the user message in the prompt
            for message in example["prompt"]:
                if message["role"] == "user":
                    # Extract the existing prompt content
                    content = message["content"]
                    # Check if content has <video> tag
                    if "<video>" in content:
                        video_question = content.replace("<video>", "").strip()
                        
                        # Extract the question text and time range
                        question_text = ""
                        time_range = example["extra_info"].get("time_range", [0, 10])
                        
                        if "Please answer the question about the video:" in video_question:
                            question_text = video_question.split("Please answer the question about the video:")[1].split("with a series")[0].strip()
                        
                        new_prompt = f"""<video>Please answer the question about the video: {question_text}
IMPORTANT: Your ENTIRE response must be ONLY a valid JSON dictionary mapping frame timestamps to bounding boxes.
For each whole second within the time range {time_range}, provide coordinates as [x1, y1, x2, y2].
Example: {{"3": [100, 150, 200, 250], "4": [105, 155, 205, 255]}}
Do not use code blocks, do not add explanations, and ensure x2>x1 and y2>y1 for all boxes."""
                        
                        message["content"] = new_prompt
                        break

    # --- Split into train/val sets ---
    random.seed(args.seed)
    random.shuffle(all_examples)

    train_val_split = int(len(all_examples) * (1 - args.val_percent))
    train_data = all_examples[:train_val_split]
    val_data = all_examples[train_val_split:]

    logger.info(f"Split data: {len(train_data)} train, {len(val_data)} validation examples.")

    # --- Define features schema ---
    # Define base features
    base_features = {
        "data_source": datasets.Value("string"),
        "prompt": [{"role": datasets.Value("string"), "content": datasets.Value("string")}],
        "videos": [{
            "type": datasets.Value("string"), 
            "video": datasets.Sequence(datasets.Value("string")),  # Now a list of frame paths
            "fps": datasets.Value("float32")
            # Removed total_pixels and min_pixels as they're no longer needed here
        }],
        "ability": datasets.Value("string"), # vstar_temporal or vstar_spatial
        "reward_model": {"ground_truth": datasets.Value("string")}, # Store GT as JSON string
        "extra_info": {
            "vid": datasets.Value("string"),
            "domain": datasets.Value("string"),
            "original_question": datasets.Value("string"),
            "task_type": datasets.Value("string"), # 'temporal' or 'spatial'
            "original_video_file": datasets.Value("string"),
            "frames_dir": datasets.Value("string")
            # Other fields like time_range are added conditionally below
        }
    }

    # Add optional video_tokens to extra_info schema if calculated
    if args.calculate_tokens and any(ex["extra_info"].get("video_tokens") for ex in all_examples):
        base_features["extra_info"]["video_tokens"] = {
            "duration": datasets.Value("float32"),
            "width": datasets.Value("int32"),
            "height": datasets.Value("int32"),
            "num_frames": datasets.Value("int32"),
            "tokens_per_frame": datasets.Value("int32"),
            "total_tokens": datasets.Value("int32")
        }

    # Add time_range to extra_info schema for spatial tasks
    # This handles both temporal and spatial examples correctly
    if args.slice_type != "temporal":
        base_features["extra_info"]["time_range"] = datasets.Sequence(datasets.Value("int32"))

    features = datasets.Features(base_features)

    # --- Create and save datasets ---
    try:
        logger.info("Creating Hugging Face Datasets...")
        train_dataset = datasets.Dataset.from_list(train_data, features=features)
        logger.info(f"Saving training dataset ({len(train_dataset)} rows) to {train_parquet_path}")
        train_dataset.to_parquet(train_parquet_path)

        if val_data:
            val_dataset = datasets.Dataset.from_list(val_data, features=features)
            logger.info(f"Saving validation dataset ({len(val_dataset)} rows) to {val_parquet_path}")
            val_dataset.to_parquet(val_parquet_path)
        else:
            logger.info("No validation data to save.")

        logger.info("Datasets saved successfully!")

        # Print summary of the frame-based preprocessing
        print("\n=== V-STAR Frame Extraction Summary ===")
        print(f"Total videos processed: {extraction_stats['total_videos']}")
        print(f"Successfully processed videos: {extraction_stats['videos_processed']}")
        print(f"Total frames extracted: {extraction_stats['frames_extracted']}")
        print(f"Failed extractions: {extraction_stats['failed_extractions']}")
        print(f"\nFrames saved to: {frames_base_dir}")
        print(f"Data saved to: {train_parquet_path} and {val_parquet_path}")
        print("\nAdvantages of frame preprocessing:")
        print(" - Significantly faster data loading during training")
        print(" - Parallelized video decoding only happens once during preprocessing")
        print(" - Consistent frame sampling across training runs")
        print("\nTo use this dataset for training:")
        print(f"  python3 -m verl.trainer.main_ppo [...] data.train_files={train_parquet_path} data.val_files={val_parquet_path} [...]")
        print("\nNote: This approach trades faster training for increased disk usage.")
        print("      The extracted frames typically use more storage than the original videos.")
        print("==================================")

    except Exception as e:
        logger.error(f"Error creating or saving datasets: {e}", exc_info=True)


if __name__ == "__main__":
    main() 