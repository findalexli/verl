#!/usr/bin/env python3
# Copyright 2024 ByteDance
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# notedowon which envieornemtn i was running
import argparse
import json
import os
import random
import time
import signal
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
from collections import Counter

import datasets
from tqdm import tqdm
import re
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)
# List of problematic video filenames to skip
# Example: PROBLEMATIC_VIDEOS_TO_SKIP = ["video1.mp4", "another_problem_video.mp4"]
# Note: Videos with malformed H.264 metadata that cause Decord warnings but are actually processable
# can be added here to avoid warning spam, or use --skip_indices to skip by index
PROBLEMATIC_VIDEOS_TO_SKIP = [
    "qQWZNLuOdLw_jumping_jacks_[10].mp4",  # Malformed H.264 metadata, but processable via fetch_video fallback
    "HdJO0vQX7q0_bouncing_on_bouncy_castle_[5].mp4",  # KeyError: 'video_fps' during training
    "AsJgDPLu_ro_battle_rope_training_[3].mp4",  # Decord error: cannot find video stream with wanted index: -1
]

# TODO: change the following to a single prompt 
INSTRUCTIONS = [
    # Detailed Instructions
    "This video shows someone doing the {movement} exercise. Please count the total number of {movement} repetitions performed. Remember that the speed might change, so count carefully. Explain your reasoning step by step and put your final answer in \\boxed{{}}.",
    "Count the total reps of the {movement} exercise in this video. Be mindful of varying speed. Provide your step-by-step reasoning and box the final count like \\boxed{{your answer}}.",
    "How many {movement} reps are completed in this video? Analyze the video carefully as the pace may vary. Give your reasoning and enclose the final count in \\boxed{{}}.",
    "Please determine the number of {movement} repetitions shown in this {movement} video. The performer's frequency might not be constant. Explain your thought process and present the final count in \\boxed{{}}.",
    "What is the total count for the {movement} exercise in this video? Count each rep individually, as the speed can change. Justify your answer step by step and use \\boxed{{}} for the final number.",
    
    # More Casual Instructions
    "Count the {movement} movements. The speed isn't constant. Explain how you counted and put the final number in \\boxed{{}}.",
    "Hey, can you count the {movement} reps here? Speed might change. Show your work and box the final answer: \\boxed{{count}}.",
    "Count the reps for {movement}. Explain your steps and put the final count in \\boxed{{}}.",
    "Tell me the rep count for {movement}. Reason it out and use \\boxed{{}} for the answer.",
    "How many times did they do the {movement}? Show reasoning and use \\boxed{{final answer}}.",
    "Count the {movement} exercise reps, explain, and box the answer \\boxed{{}}.",
]
coountix_instructions = [
    # General/Detailed Instructions
    "Count the total repetitions of the {movement} action in this video. Be mindful of varying speed. Provide your step-by-step reasoning and box the final count like \\boxed{{your answer}}.",
    "How many times is the {movement} action completed in this video? Analyze the video carefully as the pace may vary. Give your reasoning and enclose the final count in \\boxed{{}}.",
    "Please determine the number of {movement} repetitions shown in this video. The performer's frequency might not be constant. Explain your thought process and present the final count in \\boxed{{}}.",
    "What is the total count for the {movement} action in this video? Count each repetition individually, as the speed can change. Justify your answer step by step and use \\boxed{{}} for the final number.",
    "Count the repetitions of {movement}. Explain your steps and put the final count in \\boxed{{}}.",
    "Tell me the repetition count for {movement}. Reason it out and use \\boxed{{}} for the answer.",
    "How many times did they perform {movement}? Show reasoning and use \\boxed{{final answer}}.",
    "Count the {movement} action repetitions, explain, and box the answer \\boxed{{}}.",
]

def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe"""
    try:
        # Remove file:// prefix if present
        if video_path.startswith("file://"):
            video_path = video_path[7:]
            
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'json', 
            video_path
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
        data = json.loads(output)
        return float(data['format']['duration'])
    except subprocess.TimeoutExpired:
        logger.warning(f"ffprobe timed out for duration check on {video_path}")
        return 0
    except Exception as e:
        logger.debug(f"Could not get video duration for {video_path}: {e}")
        return 0

def get_video_dimensions(video_path):
    """Get video width and height using ffprobe"""
    try:
        # Remove file:// prefix if present
        if video_path.startswith("file://"):
            video_path = video_path[7:]
            
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-select_streams', 'v:0', 
            '-show_entries', 'stream=width,height', 
            '-of', 'json', 
            video_path
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
        data = json.loads(output)
        
        if 'streams' in data and len(data['streams']) > 0:
            width = int(data['streams'][0]['width'])
            height = int(data['streams'][0]['height'])
            return width, height
        return 0, 0
    except subprocess.TimeoutExpired:
        logger.warning(f"ffprobe timed out for dimension check on {video_path}")
        return 0,0
    except Exception as e:
        logger.debug(f"Could not get video dimensions for {video_path}: {e}")
        return 0, 0

def timeout_handler(signum, frame):
    raise TimeoutError("Video loading timed out")

def validate_video_with_decord_timeout(video_path, timeout_seconds=10):
    """
    Validate that a video can be loaded with decord within the timeout.
    This ensures consistency with training where decord is the primary backend.
    """
    try:
        # Remove file:// prefix if present
        if video_path.startswith("file://"):
            video_path = video_path[7:]
        
        # Check if decord is available
        try:
            import decord
        except ImportError:
            logger.warning(f"Decord not available, skipping decord validation for {video_path}")
            return True  # If decord not available, assume it's fine
        
        # Set up timeout signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            start_time = time.time()
            
            # Try to create VideoReader - this is where most hangs occur
            vr = decord.VideoReader(video_path)
            
            # Basic validation
            total_frames = len(vr)
            video_fps = vr.get_avg_fps()
            
            # Try to read first frame to ensure it's actually decodable
            if total_frames > 0:
                first_frame = vr[0]
                if first_frame is None:
                    logger.warning(f"Video {video_path} first frame is None")
                    return False
            
            elapsed = time.time() - start_time
            signal.alarm(0)  # Cancel timeout
            
            logger.debug(f"Decord validation passed for {video_path}: {total_frames} frames, {video_fps:.2f} fps, {elapsed:.3f}s")
            return True
            
        except TimeoutError:
            signal.alarm(0)  # Cancel timeout
            logger.warning(f"Decord validation timed out for {video_path} after {timeout_seconds}s")
            return False
            
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            logger.warning(f"Decord validation failed for {video_path}: {e}")
            return False
            
    except Exception as e:
        logger.warning(f"Error setting up decord validation for {video_path}: {e}")
        return False

def validate_and_calculate_video_tokens(video_path, max_video_tokens=24000):
    """
    Validate video with decord timeout first, then calculate tokens using qwen_vl_utils.
    This ensures training consistency by filtering out videos that hang with decord.
    """
    
    try:
        # Hardcoded skip for problematic videos (moved here to be checked first)
        if Path(video_path).name in PROBLEMATIC_VIDEOS_TO_SKIP:
            logger.warning(f"Hardcoded skip for problematic video: {video_path}")
            return None

        # Step 1: Perform ffprobe checks first for basic validity
        # Using actual video_path, not file:// URI for ffprobe
        duration = get_video_duration(video_path) 
        orig_width, orig_height = get_video_dimensions(video_path)

        if duration <= 0:
            logger.warning(f"Video {video_path} has invalid duration via ffprobe: {duration}s. Skipping.")
            return None
        if orig_width <= 0 or orig_height <= 0:
            logger.warning(f"Video {video_path} has invalid dimensions via ffprobe: {orig_width}x{orig_height}. Skipping.")
            return None
        
        if duration <= 2.0: 
            logger.info(f"Skipping very short video {video_path} based on ffprobe duration: {duration} seconds")
            return None

        # Step 2: NEW - Validate with decord timeout for training consistency
        logger.debug(f"Testing decord loading for {video_path}")
        if not validate_video_with_decord_timeout(video_path, timeout_seconds=10):
            logger.warning(f"Video {video_path} failed decord validation (timeout or error). Skipping for training consistency.")
            return None

        # Step 3: Proceed with Qwen VL's fetch_video for token calculation
        from qwen_vl_utils.vision_process import fetch_video
        
        # Create video config matching our training setup
        video_config = {
            "video": f"file://{video_path}",
            "total_pixels": max_video_tokens * 28 * 28,  # Same as our training config
        }
        
        # Time the video fetching process
        fetch_start_time = time.time()
        
        # Actually load the video to get the real tensor shape - this validates decoding too
        video_tensor = fetch_video(video_config)
        
        fetch_end_time = time.time()
        fetch_duration = fetch_end_time - fetch_start_time
        
        # Basic validation of the tensor result from fetch_video
        if video_tensor is None:
            logger.warning(f"fetch_video returned None tensor for {video_path}")
            return None
        
        # The output tensor shape is (num_frames, channels, height, width)
        if hasattr(video_tensor, 'shape'):
            if len(video_tensor.shape) != 4:  # Expected (num_frames, C, H, W)
                logger.warning(f"fetch_video returned tensor with invalid shape for {video_path}: {video_tensor.shape}")
                return None
            num_frames, channels, height, width = video_tensor.shape
        else:
            # Handle list of frames case
            if not video_tensor:  # Empty list
                logger.warning(f"fetch_video returned an empty list of frames for {video_path}")
                return None
            num_frames = len(video_tensor)
            if num_frames > 0:
                # Each frame is (channels, height, width)
                if hasattr(video_tensor[0], 'shape') and len(video_tensor[0].shape) != 3:  # Expected (C,H,W) for each frame
                    logger.warning(f"fetch_video returned list of frames with invalid shape for {video_path}: {video_tensor[0].shape}")
                    return None
                channels, height, width = video_tensor[0].shape
            else:
                logger.warning(f"Video {video_path} produced no frames")
                return None
        
        # Validate minimum requirements
        if num_frames < 2:  # Need at least 2 frames for meaningful repetition counting
            logger.warning(f"Video {video_path} has too few frames: {num_frames}")
            return None
            
        if height < 28 or width < 28:  # Minimum reasonable processed size
            logger.warning(f"Video {video_path} processed dimensions too small: {width}x{height}")
            return None
        
        # Calculate tokens per frame using the actual processed dimensions
        # This matches the qwen_vl_utils logic: integer division by 28
        tokens_per_frame = (height // 28) * (width // 28)
        
        # Total tokens is tokens per frame * number of frames
        total_tokens = tokens_per_frame * num_frames
        
        # Check if tokens exceed limit
        if total_tokens > max_video_tokens:
            logger.info(f"Video {video_path} exceeds token limit: {total_tokens} > {max_video_tokens}")
            return None
        
        # Skip videos that take too long to fetch (likely problematic)
        if fetch_duration > 10.0:
            logger.info(f"Video {video_path} took too long to fetch: {fetch_duration:.3f}s > 10.0s, skipping")
            return None
        
        logger.info(f"Video {video_path}: original {orig_width}x{orig_height}, processed {width}x{height}, "
                   f"frames: {num_frames}, tokens/frame: {tokens_per_frame}, total: {total_tokens}, "
                   f"fetch_time: {fetch_duration:.3f}s")
        
        return {
            'duration': duration,
            'original_width': orig_width,
            'original_height': orig_height,
            'processed_width': width,
            'processed_height': height,
            'num_frames': num_frames,
            'tokens_per_frame': tokens_per_frame,
            'total_tokens': total_tokens,
            'fetch_time': fetch_duration
        }
        
    except Exception as e:
        logger.warning(f"Video validation/token calculation failed for {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_movement_info(filename):
    # Extract movement name and side (if any)
    movement_match = re.search(r'_(.*?)(?:\((L|R|l|r)\))_', filename)
    if movement_match:
        movement = movement_match.group(1)
        side = movement_match.group(2)
        if side and side.upper() == 'L':
            movement = f'left {movement}'
        elif side and side.upper() == 'R':
            movement = f'right {movement}'
    else:
        movement = None
        
    # 检查中括号模式 [L|R]
    bracket_match = re.search(r'_(.*?)\[(L|R|l|r)\]_', filename)
    if not movement and bracket_match:
        movement = bracket_match.group(1)
        side = bracket_match.group(2)
        if side and side.upper() == 'L':
            movement = f'left {movement}'
        elif side and side.upper() == 'R':
            movement = f'right {movement}'
    
    # 如果两种模式都没匹配到，再尝试匹配没有方向的动作名称
    if not movement:
        basic_match = re.search(r'_(.*?)_', filename)
        if basic_match:
            movement = basic_match.group(1)

    # Extract ground truth numbers
    ground_truth_match = re.search(r'\[(\d+(?:-\d+)*)\]', filename)
    if ground_truth_match:
        try:
            # Convert to integer if it's a single number, otherwise keep as string
            ground_truth = ground_truth_match.group(1)
            if '-' not in ground_truth:
                ground_truth = int(ground_truth)
        except ValueError:
            ground_truth = None
    else:
        ground_truth = None

    return movement, ground_truth
    

def resize_video(video_path, output_path=None, target_height=480):
    """Resize a video to a target height (480p) while maintaining aspect ratio"""
    try:
        # If no output path specified, create one with _480p suffix
        if output_path is None:
            video_path_obj = Path(video_path)
            output_path = str(video_path_obj.parent / f"{video_path_obj.stem}_480p{video_path_obj.suffix}")
        
        # Remove file:// prefix if present
        if video_path.startswith("file://"):
            video_path = video_path[7:]
        
        # Get original dimensions
        width, height = get_video_dimensions(video_path)
        if height <= 0 or width <= 0:
            logger.warning(f"Could not get dimensions for {video_path}")
            return None
        
        # Skip resizing if video is already smaller than or equal to target height
        if height <= target_height:
            logger.info(f"Skipping resize for {video_path} - original height {height} is already <= target height {target_height}")
            return video_path  # Return original path since no resize needed
            
        # Calculate new width while maintaining aspect ratio
        new_width = int((width / height) * target_height)
        new_width = new_width - (new_width % 2)  # Ensure even width for compatibility
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'scale={new_width}:{target_height}',
            '-c:v', 'libx264',
            '-crf', '23',  # Balance quality and file size
            '-preset', 'medium',  # Encoding speed/compression ratio
            '-c:a', 'copy',  # Copy audio stream without re-encoding
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        
        logger.info(f"Resizing video {video_path} to {new_width}x{target_height}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        
        return output_path
    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg resize timed out for {video_path}")
        return None
    except Exception as e:
        logger.error(f"Error resizing video {video_path}: {e}")
        return None

def process_single_video(video_info):
    """Process a single video file and return the example dict or None if invalid."""
    clip_file, folder_path, data_source, resize_videos, max_video_tokens = video_info
    
    # Hardcoded skip for problematic videos
    if clip_file.name in PROBLEMATIC_VIDEOS_TO_SKIP:
        logger.warning(f"Hardcoded skip for problematic video: {clip_file.name}")
        return None

    clip_name = clip_file.name
    
    # Extract exercise info from filename
    exercise_name, rep_count = extract_movement_info(clip_name)
    if rep_count is None:
        return None

    # Format as absolute path for video processor
    original_video_path = str(clip_file.absolute())
    
    # Resize video if requested
    video_path = original_video_path
    if resize_videos:
        resized_clips_dir = folder_path / "clips_480p"
        os.makedirs(resized_clips_dir, exist_ok=True)
        resized_video_path = str(resized_clips_dir / f"{clip_file.stem}_480p{clip_file.suffix}")
        
        # Check if the resized video exists and is valid
        should_resize = True
        if os.path.exists(resized_video_path):
            try:
                # Use ffprobe to quickly check if the file is a valid video
                ffprobe_cmd = [
                    'ffprobe',
                    '-v', 'error',  # Only show errors
                    '-select_streams', 'v:0', # Check only the first video stream
                    '-show_entries', 'stream=codec_name', # Request minimal info
                    '-of', 'default=nokey=1:noprint_wrappers=1', # Minimal output format
                    resized_video_path
                ]
                result = subprocess.run(ffprobe_cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    should_resize = False # File exists and ffprobe succeeded, skip resizing
                    logger.info(f"Resized video already exists and is valid: {resized_video_path}. Skipping resize.")
                else:
                    logger.warning(f"Resized video exists but seems invalid (ffprobe failed): {resized_video_path}. Will attempt resize.")
            except Exception as e:
                logger.warning(f"Error running ffprobe check on {resized_video_path}: {e}. Will attempt resize.")

        if should_resize:
            logger.info(f"Attempting to resize {original_video_path} to {resized_video_path}")
            resized_path_result = resize_video(original_video_path, resized_video_path)
            # Check if resizing was successful before assigning path
            if resized_path_result and os.path.exists(resized_path_result):
                video_path = resized_path_result
            else:
                logger.error(f"Resizing failed for {original_video_path}, using original.")
                # Stick with original_video_path if resizing failed
                video_path = original_video_path
        else:
            # Use the existing valid resized video path
            video_path = resized_video_path

    # Format as file:// URL for video processor
    video_url = f"file://{video_path}"
    
    # Validate video and calculate tokens
    video_tokens_info = validate_and_calculate_video_tokens(video_path, max_video_tokens)
    
    # Skip if video validation failed or tokens exceed limit
    if video_tokens_info is None:
        return None
        
    # Create the prompt with a random instruction template formatted with the movement name
    instruction_template = random.choice(coountix_instructions)
    instruction = instruction_template.format(movement=exercise_name)
    
    # Format the example
    example = {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": f"<video>{instruction}",
            }
        ],
        "videos": [{
            "type": "video",
            "video": video_url,
            # Remove fps to use default FPS=2.0, let vision processor optimize frame sampling
            "total_pixels": max_video_tokens * 28 * 28,  # Main constraint: total token budget
            "do_rescale": False  # Avoid double rescaling if already in [0, 1]
        }],
        "ability": "exercise_rep_counting",
        "reward_model": {
            "style": "rule",
            "ground_truth": str(rep_count)
        },
        "extra_info": {
            "exercise_name": exercise_name,
            "video_file": video_path,
            "original_video_file": original_video_path if resize_videos else video_path,
            "rep_count": rep_count,
            "video_tokens": video_tokens_info
        },
    }
    
    return example

def collect_all_videos(data_dir: Path, data_source: str, resize_videos: bool, 
                       max_video_tokens: int, max_samples: int = None, skip_indices: List[int] = None) -> List[Dict]:
    """Process all videos in all folders using a single loop for maximum efficiency."""
    all_examples = []
    video_count = 0
    skip_indices = skip_indices or []  # Default to empty list if None
    
    # Find all video folders
    video_folders = [f for f in data_dir.iterdir() if f.is_dir() and f.name not in ['.cache']]
    print(f"Found {len(video_folders)} folders: {[f.name for f in video_folders]}")
    if skip_indices:
        print(f"Will skip videos at indices: {skip_indices}")
    
    # Process each folder
    for folder_path in video_folders:
        # Load metadata to ensure folder is valid
        metadata_path = folder_path / "metadata.json"
        if not metadata_path.exists():
            print(f"Metadata not found in {folder_path}, skipping")
            continue
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            print(f"Error parsing metadata in {folder_path}, skipping")
            continue
        
        # Process clips directory
        clips_dir = folder_path / "clips"
        if not clips_dir.exists() or not clips_dir.is_dir():
            print(f"Clips directory not found in {folder_path}, skipping")
            continue
        
        # Find all mp4 files in the clips directory
        clip_files = list(clips_dir.glob("*.mp4"))
        print(f"Processing folder {folder_path.name}: found {len(clip_files)} videos")
        
        # Process each video in this folder
        successful = 0
        failed = 0
        skipped_by_index = 0
        
        with tqdm(total=len(clip_files), desc=f"{folder_path.name}") as pbar:
            for clip_file in clip_files:
                # Exit early if we've reached max samples
                if max_samples and len(all_examples) >= max_samples:
                    print(f"Reached max samples limit ({max_samples}), stopping")
                    break
                
                # Check if this video index should be skipped
                if video_count in skip_indices:
                    logger.info(f"Skipping video at index {video_count}: {clip_file.name}")
                    skipped_by_index += 1
                    video_count += 1
                    pbar.update(1)
                    continue
                
                # Process single video
                video_info = (clip_file, folder_path, data_source, resize_videos, max_video_tokens)
                try:
                    example = process_single_video(video_info)
                    
                    if example is not None:
                        all_examples.append(example)
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    # This catches errors in the loop itself, not in process_single_video
                    print(f"Critical error in main loop for {clip_file.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
                
                # Update progress
                video_count += 1
                pbar.update(1)
                
                # Show stats periodically
                if video_count % 50 == 0:
                    pbar.set_postfix({
                        'folder_success': successful, 
                        'folder_failed': failed,
                        'skipped_idx': skipped_by_index,
                        'total_collected': len(all_examples),
                        'rate': f"{successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "0%"
                    })
        
        print(f"Folder {folder_path.name} done: {successful} successful, {failed} failed, {skipped_by_index} skipped by index.")
        
        # Exit early if we've reached max samples
        if max_samples and len(all_examples) >= max_samples:
            all_examples = all_examples[:max_samples]  # Ensure exactly max_samples
            break
    
    print(f"Total videos processed: {video_count}")
    print(f"  - Successful: {len(all_examples)}")
    print(f"  - Failed: {video_count - len(all_examples) - len([i for i in skip_indices if i < video_count])}")
    print(f"  - Skipped by index: {len([i for i in skip_indices if i < video_count])}")
    if video_count > 0:
        print(f"  - Success rate: {len(all_examples)/video_count*100:.1f}%")
    return all_examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="Path to the directory containing video folders",
    )
    parser.add_argument(
        "--output_dir",
        help="Local directory to save the output parquet files",
    )
    parser.add_argument(
        "--train_filename",
        default="train.parquet",
        help="Name for the training output parquet file",
    )
    parser.add_argument(
        "--val_filename",
        default="val.parquet",
        help="Name for the validation output parquet file",
    )
    parser.add_argument(
        "--val_percent",
        type=float,
        default=0.1,
        help="Percentage of data to use for validation (between 0 and 1)",
    )
    parser.add_argument(
        "--data_source_name",
        default="repcount",
        help="Name to use as the data_source identifier",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum total number of samples to process (for testing)",
    )
    parser.add_argument(
        "--max_videos_to_process",
        type=int,
        default=None,
        help="Maximum total number of videos to process (stops after this many videos regardless of success/failure)",
    )
    parser.add_argument(
        "--resize_videos",
        action="store_true",
        help="Resize videos to 480p",
    )
    parser.add_argument(
        "--token_stats_file",
        default="video_token_stats.json",
        help="File to save token statistics to",
    )
    parser.add_argument(
        "--max_video_tokens",
        type=int,
        default=24000,
        help="Maximum video tokens allowed (leaving buffer for text prompt)",
    )
    parser.add_argument(
        "--skip_indices",
        type=str,
        default="",
        help="Comma-separated list of video indices to skip (e.g., '2588,1234,5678')",
    )

    args = parser.parse_args()

    # Parse skip_indices
    skip_indices = []
    if args.skip_indices:
        try:
            skip_indices = [int(x.strip()) for x in args.skip_indices.split(',') if x.strip()]
            print(f"Will skip videos at indices: {skip_indices}")
        except ValueError as e:
            print(f"Error parsing skip_indices: {e}")
            print("skip_indices should be comma-separated integers (e.g., '2588,1234,5678')")
            return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    train_parquet_path = os.path.join(args.output_dir, args.train_filename)
    val_parquet_path = os.path.join(args.output_dir, args.val_filename)

    # Process all videos in a single loop
    print(f"Starting video processing with max_samples={args.max_samples if args.max_samples else 'all'}")
    all_examples = collect_all_videos(
        Path(args.data_dir), 
        args.data_source_name, 
        args.resize_videos, 
        args.max_video_tokens,
        args.max_samples,
        skip_indices
    )
    
    if not all_examples:
        print("No valid examples were processed. Exiting.")
        return
    
    # Calculate token distribution (we always have this now)
    print("Calculating video token distribution...")
    token_counts = []
    frame_counts = []
    duration_counts = []
    tokens_per_frame_counts = []
    
    print(f"Processing {len(all_examples)} examples for statistics...")
    for i, example in enumerate(all_examples):
        if i % 500 == 0:  # Progress indicator every 500 examples
            print(f"  Processed {i}/{len(all_examples)} examples for stats...")
        
        if "video_tokens" in example["extra_info"]:
            token_info = example["extra_info"]["video_tokens"]
            token_counts.append(token_info["total_tokens"])
            frame_counts.append(token_info["num_frames"])
            duration_counts.append(token_info["duration"])
            tokens_per_frame_counts.append(token_info["tokens_per_frame"])
        
    print("Computing statistical summaries...")
    # Calculate statistics
    stats = {
        "token_stats": {
            "min": min(token_counts) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
            "mean": sum(token_counts) / len(token_counts) if token_counts else 0,
            "median": sorted(token_counts)[len(token_counts)//2] if token_counts else 0,
            "p95": sorted(token_counts)[int(len(token_counts)*0.95)] if token_counts and len(token_counts)*0.95 < len(token_counts) else 0,
            "p99": sorted(token_counts)[int(len(token_counts)*0.99)] if token_counts and len(token_counts)*0.99 < len(token_counts) else 0,
            "distribution": Counter([t // 1000 * 1000 for t in token_counts]) if token_counts else Counter()
        },
        "frame_stats": {
            "min": min(frame_counts) if frame_counts else 0,
            "max": max(frame_counts) if frame_counts else 0,
            "mean": sum(frame_counts) / len(frame_counts) if frame_counts else 0,
            "distribution": Counter(frame_counts) if frame_counts else Counter()
        },
        "duration_stats": {
            "min": min(duration_counts) if duration_counts else 0,
            "max": max(duration_counts) if duration_counts else 0,
            "mean": sum(duration_counts) / len(duration_counts) if duration_counts else 0,
            "distribution": Counter([int(d) for d in duration_counts]) if duration_counts else Counter()
        },
        "tokens_per_frame_stats": {
            "min": min(tokens_per_frame_counts) if tokens_per_frame_counts else 0,
            "max": max(tokens_per_frame_counts) if tokens_per_frame_counts else 0,
            "mean": sum(tokens_per_frame_counts) / len(tokens_per_frame_counts) if tokens_per_frame_counts else 0,
            "distribution": Counter(tokens_per_frame_counts) if tokens_per_frame_counts else Counter()
        }
    }
    
    # Save statistics to file
    print("Saving token statistics...")
    stats_file = os.path.join(args.output_dir, args.token_stats_file)
    with open(stats_file, 'w') as f:
        # Convert Counter objects to dictionaries for JSON serialization
        for category in stats:
            if "distribution" in stats[category]:
                stats[category]["distribution"] = dict(stats[category]["distribution"])
        
        json.dump(stats, f, indent=2)
    
    print(f"Token statistics saved to {stats_file}")
    print(f"Token stats summary:")
    if token_counts: # Ensure not printing stats for empty lists
        print(f"  Min tokens: {stats['token_stats']['min']}")
        print(f"  Max tokens: {stats['token_stats']['max']}")
        print(f"  Mean tokens: {stats['token_stats']['mean']:.2f}")
        print(f"  Median tokens: {stats['token_stats']['median']}")
        print(f"  95th percentile: {stats['token_stats']['p95']}")
        print(f"  99th percentile: {stats['token_stats']['p99']}")

    # Split into train/val sets
    print("Splitting data into train/val sets...")
    random.seed(42)  # For reproducibility
    random.shuffle(all_examples)
    
    train_val_split = int(len(all_examples) * (1 - args.val_percent))
    train_data = all_examples[:train_val_split]
    val_data = all_examples[train_val_split:]
    
    print(f"Split data into {len(train_data)} training examples and {len(val_data)} validation examples")
    
    # Define features schema for the datasets
    print("Defining dataset schema...")
    features = datasets.Features({
        "data_source": datasets.Value("string"),
        "prompt": [
            {
                "role": datasets.Value("string"),
                "content": datasets.Value("string")
            }
        ],
        "videos": [
            {
                "type": datasets.Value("string"),
                "video": datasets.Value("string"),
                "total_pixels": datasets.Value("int32"),
                "do_rescale": datasets.Value("bool")
            }
        ],
        "ability": datasets.Value("string"),
        "reward_model": {
            "style": datasets.Value("string"),
            "ground_truth": datasets.Value("string")
        },
        "extra_info": {
            "exercise_name": datasets.Value("string"),
            "video_file": datasets.Value("string"),
            "original_video_file": datasets.Value("string"),
            "rep_count": datasets.Value("int32"),
            "video_tokens": {
                "duration": datasets.Value("float32"),
                "original_width": datasets.Value("int32"),
                "original_height": datasets.Value("int32"),
                "processed_width": datasets.Value("int32"),
                "processed_height": datasets.Value("int32"),
                "num_frames": datasets.Value("int32"),
                "tokens_per_frame": datasets.Value("int32"),
                "total_tokens": datasets.Value("int32"),
                "fetch_time": datasets.Value("float32")
            }
        }
    })
    
    # Create and save datasets
    try:
        print(f"Creating training dataset from {len(train_data)} examples...")
        print("  This may take several minutes for large datasets...")
        train_dataset = datasets.Dataset.from_list(train_data, features=features)
        
        print(f"Saving training dataset to {train_parquet_path}")
        print("  Writing parquet file (this may take a while)...")
        train_dataset.to_parquet(train_parquet_path)
        print("  Training dataset saved successfully!")
        
        print(f"Creating validation dataset from {len(val_data)} examples...")
        val_dataset = datasets.Dataset.from_list(val_data, features=features)
        
        print(f"Saving validation dataset to {val_parquet_path}")
        print("  Writing parquet file...")
        val_dataset.to_parquet(val_parquet_path)
        print("  Validation dataset saved successfully!")
        
        print("All datasets saved successfully!")
        
    except Exception as e:
        print(f"Error saving datasets: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()