#!/usr/bin/env python3

import pandas as pd
import random
from pathlib import Path
import sys
import os

# Single detailed prompt template for all examples
SINGLE_DETAILED_PROMPT = """This video shows someone performing the {movement} exercise. Please provide a detailed scene-by-scene, timestamp-by-timestamp analysis of the video to count the repetitions accurately.

1. First, describe what you observe in the video chronologically, noting key timestamps when repetitions begin and end
2. For each repetition, describe the specific body movements and positions that constitute one complete cycle
3. Count each repetition systematically, explaining your reasoning for what constitutes a complete rep
4. Ignore any visual counters, timers, or text overlays in the video - focus only on the actual physical exercise movements
5. Provide your final count in \\boxed{{}}

Be thorough in your analysis and explain your counting methodology step by step."""

def update_parquet_with_single_prompt(input_file, output_file):
    """Update parquet file with single detailed prompt template"""
    
    print(f"Loading {input_file}...")
    
    try:
        df = pd.read_parquet(input_file)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return False
    
    print(f"Original dataset has {len(df)} samples")
    
    updated_samples = []
    
    for idx, row in df.iterrows():
        # Extract exercise name from extra_info
        exercise_name = row['extra_info']['exercise_name']
        
        # Use the single detailed prompt template
        new_instruction = SINGLE_DETAILED_PROMPT.format(movement=exercise_name)
        
        # Update the prompt content while keeping the same structure
        updated_row = row.copy()
        updated_row['prompt'] = [
            {
                "role": "user", 
                "content": f"<video>{new_instruction}"
            }
        ]
        
        updated_samples.append(updated_row)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} samples...")
    
    # Create new dataframe
    updated_df = pd.DataFrame(updated_samples)
    
    print(f"Saving updated dataset to {output_file}...")
    try:
        updated_df.to_parquet(output_file)
        print(f"Successfully updated {len(updated_df)} samples")
        return True
    except Exception as e:
        print(f"Error saving {output_file}: {e}")
        return False

def show_examples(df, filename, num_examples=2):
    """Show example prompts from the updated dataset"""
    print(f"\nExample prompts from {filename}:")
    for i in range(min(num_examples, len(df))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Exercise: {df.iloc[i]['extra_info']['exercise_name']}")
        print(f"Ground truth: {df.iloc[i]['extra_info']['rep_count']}")
        print(f"Video file: {Path(df.iloc[i]['extra_info']['video_file']).name}")
        prompt_content = df.iloc[i]['prompt'][0]['content']
        # Show first 200 chars of prompt
        print(f"Prompt preview: {prompt_content[:200]}...")
