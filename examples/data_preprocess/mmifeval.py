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

import argparse
import base64 # For checking if image is base64
import json
import logging
import os
import random
import sys
import time
import ast # For safer parsing of problematic strings
import numpy as np # For handling numpy arrays in data
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
from io import BytesIO

import datasets
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# Define the schema for the output Parquet file (focused on 'main' inference type)
# This schema should contain all info needed for the reward model, except aux_data_dict
SCHEMA = datasets.Features({
    "id": datasets.Value("string"),
    "data_source": datasets.Value("string"),
    "prompt": [
        {
            "role": datasets.Value("string"),
            "content": datasets.Value("string")
        }
    ],
    "tag": datasets.Value("string"), # P-Level or C-Level
    # Define images using the datasets.Image() feature type within a Sequence
    "images": datasets.Sequence(datasets.Image()),
    "ability": datasets.Value("string"), # Specific ability identifier, e.g., "mmifeval"
    "ground_truth": datasets.Value("string"), # JSON string with answer or constraints for scoring
    "reward_model": {
        # For P-Level
        "answer": datasets.Sequence(datasets.Value("string"), length=-1), # List of strings for P-Level
        # For C-Level 
        "constraints": datasets.Sequence(
            {
                "key": datasets.Value("string"),
                "value": datasets.Value("string"),
                "judge": {
                    "method": datasets.Value("string")
                }
            }, length=-1)
    },
    "extra_info": {
         "original_question": datasets.Value("string"), # Raw question text
         # Optional additional fields
         "aux_info": datasets.Value("string"),
         "infer_type": datasets.Value("string"),
         "split": datasets.Value("string")
    }
})

# NumPy JSON encoder to handle arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def safe_json_dumps(obj):
    """Safely dump object to JSON string, handling NumPy arrays and other special types."""
    try:
        return json.dumps(obj, cls=NumpyEncoder)
    except TypeError as e:
        # If there are still non-serializable objects, convert them to strings
        if isinstance(obj, dict):
            converted = {}
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    converted[k] = safe_json_dumps(v)
                else:
                    try:
                        # Try regular JSON serialization first
                        json.dumps(v)
                        converted[k] = v
                    except:
                        # Fall back to string representation
                        converted[k] = str(v)
            return json.dumps(converted)
        elif isinstance(obj, list):
            converted = []
            for v in obj:
                if isinstance(v, (dict, list)):
                    converted.append(safe_json_dumps(v))
                else:
                    try:
                        # Try regular serialization
                        json.dumps(v)
                        converted.append(v)
                    except:
                        # Fall back to string
                        converted.append(str(v))
            return json.dumps(converted)
        else:
            # Last resort: convert to string
            return json.dumps(str(obj))

def process_complex_json(json_str):
    """Process complex JSON strings, prioritizing ast.literal_eval for non-standard formats."""
    if not json_str or not isinstance(json_str, str):
        return json_str # Return original if not a non-empty string

    # 1. Try standard JSON parsing first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 2. Try ast.literal_eval for Python literal syntax
        try:
            # Replace problematic escape sequences if necessary before literal_eval
            # Common issue: \\\" inside the string literal
            eval_str = json_str.replace('\\\\"', '\\"') 
            return ast.literal_eval(eval_str)
        except (ValueError, SyntaxError, MemoryError):
             # MemoryError can happen with deeply nested structures
            # 3. Try fixing common JSON escape issues as a fallback
            try:
                # Handle potential double encoding or excessive quoting
                if json_str.startswith('"') and json_str.endswith('"'):
                    unquoted = json.loads(json_str) # Let json handle unquoting
                    if isinstance(unquoted, str): # If it's still a string, try parsing again
                         return json.loads(unquoted)
                    else:
                         return unquoted # It was likely a double-encoded object
            except:
                pass # Ignore errors here and proceed

            # Try manual escape fixes
            fixes = [
                lambda s: s.replace('\\"', '"'), # Incorrect quote escaping
                lambda s: s.replace("\\\\", "\\"), # Double backslashes
                lambda s: s.replace("\\n", "\n"), # Newlines
                # Add more fixes if other patterns are identified
            ]
            
            fixed_str = json_str
            for fix in fixes:
                try:
                    fixed_str = fix(fixed_str)
                    return json.loads(fixed_str)
                except json.JSONDecodeError:
                    continue # Try the next fix

    # If all parsing fails, return original string and log a warning
    logger.warning(f"Failed to parse complex JSON string after multiple attempts: {json_str[:100]}...")
    return json_str # Return the original problematic string

def setup_file_logging(log_file_path: str):
    """Set up file logging in addition to console logging."""
    # Create a file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    logger.info(f"File logging initialized. Writing to {log_file_path}")

def validate_base64_image(base64_str: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validates base64-encoded image data.
    
    Args:
        base64_str: The base64 string to validate, with or without URI prefix
        
    Returns:
        Tuple containing:
            - bool: Whether the image is valid
            - str: The properly formatted image URI (if valid)
            - str: Failure reason (if invalid)
    """
    # Quick check for empty or non-string input
    if not base64_str or not isinstance(base64_str, str):
        return False, None, "Empty or non-string image data"
    
    # Extract the base64 part if data URI format is used
    base64_part = base64_str
    if "," in base64_str and base64_str.startswith("data:"):
        # Already has data URI prefix
        mime_type = base64_str.split(",")[0]
        base64_part = base64_str.split(",", 1)[1]
    else:
        # No prefix, we'll add one later if valid
        mime_type = None

    # Check if the string is valid base64
    if not is_base64(base64_part):
        return False, None, "Invalid base64 encoding"
    
    # Decode and check size
    try:
        decoded = base64.b64decode(base64_part)
        image_size_mb = len(decoded) / (1024 * 1024)
        
        # Check file size (max 20MB)
        if image_size_mb > 20:
            return False, None, f"Image too large ({image_size_mb:.2f} MB > 20 MB)"

        # Construct proper data URI format if needed
        if not mime_type:
            # No MIME type provided, default to JPEG
            formatted_image = f"data:image/jpeg;base64,{base64_part}"
        else:
            # Keep original MIME type but ensure format is correct
            formatted_image = base64_str
            
        return True, formatted_image, None
    
    except Exception as e:
        return False, None, f"Error processing image: {str(e)}"

def is_base64(s: str) -> bool:
    """
    Check if a string is valid base64.
    Args:
        s: String to check
    Returns:
        bool: True if valid base64, False otherwise
    """
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode() 
    except Exception:
        return False

def process_item(item_tuple: tuple) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Processes a single row (item) from the MM-IFEval dataframe.
    Filters for 'main' inference type and formats for the parquet schema.

    Args:
        item_tuple: A tuple containing (index, row_dict) where row_dict is a dictionary of row values.

    Returns:
        A tuple containing:
            - Dictionary conforming to SCHEMA if it's a valid 'main' item, else None
            - Dictionary with filtering info if item was filtered, else None
    """
    index, row = item_tuple
    try:
        # --- Filter for 'main' inference type ---
        # Adjust 'infer_type' column name if different in the actual TSV
        infer_type = row.get("infer_type", "main") # Default to 'main' if column missing
        if infer_type != "main":
            return None, {
                "id": row.get("id", f"unknown_{index}"),
                "reason": "wrong_infer_type",
                "details": f"Expected 'main', got '{infer_type}'",
                "index": index,
                "recoverable": False
            }
        
        # --- Extract required fields ---
        item_id = row.get("id")
        tag = row.get("tag") # P-Level or C-Level
        question = row.get("question")
        # Read from the original 'image' column in the TSV
        image = row.get("image") # Base64 encoded image string
        
        # Check for required fields
        if not item_id:
            return None, {"id": f"missing_id_{index}", "reason": "missing_id", "details": "No ID field", "index": index, "recoverable": False}
        if not tag:
            return None, {"id": item_id, "reason": "missing_tag", "details": "No tag field", "index": index, "recoverable": False}
        if not question:
            return None, {"id": item_id, "reason": "missing_question", "details": "No question field", "index": index, "recoverable": False}
        
        # --- Process image ---
        if not image:
            return None, {"id": item_id, "reason": "missing_image", "details": "No image field", "index": index, "recoverable": False}
        
        # Validate and format the base64 image string directly
        is_valid, formatted_image, error_reason = validate_base64_image(image)
        if not is_valid:
            return None, {
                "id": item_id, 
                "reason": "invalid_image", 
                "details": error_reason,
                "index": index,
                "recoverable": True,
                "original_image_length": len(image) if isinstance(image, str) else 0
            }
        
        # Decode image to PIL *before* creating processed_item
        pil_image = None
        try:
            # Assuming formatted_image is the base64 data URI string
            if formatted_image.startswith("data:image") and ',' in formatted_image:
                 base64_data = formatted_image.split(',', 1)[1]
            else:
                 base64_data = formatted_image # Assume raw base64 if no prefix
            image_bytes = base64.b64decode(base64_data)
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
             print(f"Error decoding image for {item_id}: {e}")
             # Decide how to handle decode error - skip item or use placeholder?
             # For now, let's skip if decoding fails
             return None, {"id": item_id, "reason": "image_decode_error", "details": str(e), "index": index, "recoverable": False}
        
        # --- Calculate and check token length ---
        # Estimate text tokens (simple approximation)
        text_tokens = len(question) // 4 if question else 0
        # Calculate image tokens based on user-provided logic
        image_tokens = (pil_image.height // 28) * (pil_image.width // 28) if pil_image else 0
        total_tokens = text_tokens + image_tokens
        
        # Filter if total tokens exceed the limit (e.g., 6000)
        MAX_TOKENS = 6000 
        if total_tokens > MAX_TOKENS:
             return None, {
                 "id": item_id,
                 "reason": "prompt_too_long",
                 "details": f"Total estimated tokens ({total_tokens} = {text_tokens} text + {image_tokens} image) exceed limit {MAX_TOKENS}",
                 "index": index,
                 "recoverable": False
             }

        # --- Process answer field (P-Level) ---
        answer = row.get("answer")
        processed_answer = None
        
        if tag == "P-Level":
            if not answer:
                return None, {"id": item_id, "reason": "missing_answer", "details": "P-Level item with no answer field", "index": index, "recoverable": False}
            
            # If answer is a string that needs to be parsed as JSON, parse it
            if isinstance(answer, str):
                try:
                    # Special handling for double-escaped JSON strings in P-Level answers
                    # Pattern: "\"[\"\"value1\"\", \"\"value2\"\"]\""
                    if answer.startswith('"[') and answer.endswith(']"'):
                        # First, remove outer quotes and unescape once
                        unescaped = answer[1:-1].replace('\\"', '"')
                        # Then parse the resulting JSON
                        processed_answer = json.loads(unescaped)
                    # Handle other patterns of escaped strings
                    elif answer.startswith('"') and answer.endswith('"') and '\\' in answer:
                        # Try to unescape once
                        try:
                            unescaped = json.loads(f"{{{answer}}}")[""]
                            processed_answer = json.loads(unescaped)
                        except:
                            # Fallback to the enhanced parser
                            processed_answer = process_complex_json(answer)
                    else:
                        # Standard JSON parsing
                        processed_answer = json.loads(answer)
                except json.JSONDecodeError as e:
                    # If all other methods fail, try the most aggressive approach
                    try:
                        # Remove all escaping and try to reconstruct the array
                        clean_answer = answer.replace('"', '').replace('\\', '')
                        if clean_answer.startswith('[') and clean_answer.endswith(']'):
                            items = clean_answer[1:-1].split(',')
                            processed_answer = [item.strip() for item in items]
                        else:
                            processed_answer = [clean_answer]
                    except Exception as e2:
                        return None, {
                            "id": item_id, 
                            "reason": "invalid_answer_json", 
                            "details": f"Error parsing answer JSON: {str(e)}",
                            "index": index,
                            "recoverable": True,
                            "answer": answer[:100] + "..." if len(answer) > 100 else answer
                        }
            else:
                # If it's already a Python object (e.g., list), use as is
                processed_answer = answer
            
            # Ensure the processed_answer is a list
            if not isinstance(processed_answer, list):
                if processed_answer is not None:
                    # Wrap non-list values in a list
                    processed_answer = [processed_answer]
                else:
                    return None, {
                        "id": item_id, 
                        "reason": "invalid_answer_format", 
                        "details": f"Expected answer to be a list, got {type(processed_answer)}",
                        "index": index,
                        "recoverable": True,
                        "answer": str(processed_answer)[:100] + "..." if len(str(processed_answer)) > 100 else str(processed_answer)
                    }
        
        # --- Process constraints (C-Level) ---
        constraints = row.get("constraints")
        processed_constraints = None

        if tag == "C-Level":
            if constraints is None:
                return None, {"id": item_id, "reason": "missing_constraints", "details": "C-Level item with no constraints field", "index": index, "recoverable": False}
            
            # Parse constraints if needed
            if isinstance(constraints, str):
                try:
                    # Use the enhanced parser for complex JSON
                    processed_constraints = process_complex_json(constraints)
                except Exception as e:
                    return None, {
                        "id": item_id, 
                        "reason": "invalid_constraints_json", 
                        "details": f"Error parsing constraints JSON: {str(e)}",
                        "index": index,
                        "recoverable": True,
                        "constraints": constraints[:100] + "..." if len(constraints) > 100 else constraints
                    }
            else:
                # Already a Python object
                processed_constraints = constraints
                
            # Handle NumPy arrays or numbers within the parsed constraints
            if isinstance(processed_constraints, np.ndarray):
                processed_constraints = processed_constraints.tolist()
            elif isinstance(processed_constraints, list):
                 # Ensure sub-elements are also serializable (convert numpy numbers)
                 processed_constraints = convert_numpy_numbers(processed_constraints)
            elif isinstance(processed_constraints, dict):
                 processed_constraints = convert_numpy_numbers(processed_constraints)
            elif isinstance(processed_constraints, (np.integer, np.floating)):
                 # Handle top-level numpy number if constraints is just that
                 processed_constraints = convert_numpy_numbers(processed_constraints)

            # Ensure it's a list structure after potential numpy conversion
            if not isinstance(processed_constraints, list):
                # Check if it's a dict with a 'value' key that contains constraints
                if isinstance(processed_constraints, dict) and "value" in processed_constraints:
                    constraint_values = processed_constraints.get("value", [])
                    if isinstance(constraint_values, np.ndarray):
                        constraint_values = constraint_values.tolist()
                        
                    if isinstance(constraint_values, list):
                        # Create properly formatted constraints
                        processed_constraints = [
                            {
                                "key": f"constraint_{i+1}",
                                "value": str(c),
                                "judge": {"method": "direct_gpt"}
                            }
                            for i, c in enumerate(constraint_values)
                        ]
                    else:
                        # Single value in the 'value' field - wrap it
                        processed_constraints = [{
                            "key": "constraint_1",
                            "value": str(constraint_values),
                            "judge": {"method": "direct_gpt"}
                        }]
                # Check for other list values in the dictionary
                elif isinstance(processed_constraints, dict) and any(isinstance(v, list) for v in processed_constraints.values()):
                    for k, v in processed_constraints.items():
                        if isinstance(v, list):
                            # Found a list value, use it
                            constraint_values = v
                            processed_constraints = [
                                {
                                    "key": f"constraint_{i+1}",
                                    "value": str(c),
                                    "judge": {"method": "direct_gpt"}
                                }
                                for i, c in enumerate(constraint_values)
                            ]
                            break
                else:
                    # Final fallback: wrap whatever we have in a list
                    try:
                        constraint_value = str(processed_constraints) 
                        processed_constraints = [{
                            "key": "constraint_1",
                            "value": constraint_value,
                            "judge": {"method": "direct_gpt"}
                        }]
                    except:
                        # If all else fails, use an empty list
                        processed_constraints = []
                        logger.warning(f"Failed to process constraints for {item_id}, using empty list. Original type: {type(row.get('constraints'))}, Value: {str(row.get('constraints'))[:100]}")
            
            # If we have a list but elements aren't properly formatted, fix them
            # Also ensure no numpy numbers remain
            if isinstance(processed_constraints, list):
                formatted_constraints = []
                for i, constraint in enumerate(processed_constraints):
                    # Ensure constraint itself and its contents are serializable
                    safe_constraint = convert_numpy_numbers(constraint)

                    if isinstance(safe_constraint, dict) and "value" in safe_constraint:
                        # Already has 'value' - just ensure proper format
                        formatted_constraint = {
                            "key": str(safe_constraint.get("key", f"constraint_{i+1}")),
                            "value": str(safe_constraint["value"]), # Ensure value is string
                            "judge": safe_constraint.get("judge", {"method": "direct_gpt"})
                        }
                        # Ensure judge is a dict
                        if not isinstance(formatted_constraint["judge"], dict):
                             formatted_constraint["judge"] = {"method": "direct_gpt"}
                        # Ensure judge method is string
                        formatted_constraint["judge"]["method"] = str(formatted_constraint["judge"].get("method", "direct_gpt"))
                        formatted_constraints.append(formatted_constraint)
                    elif isinstance(safe_constraint, str) or isinstance(safe_constraint, (int, float, bool)):
                        # String/primitive value - wrap properly
                        formatted_constraints.append({
                            "key": f"constraint_{i+1}",
                            "value": str(safe_constraint),
                            "judge": {"method": "direct_gpt"}
                        })
                    elif isinstance(safe_constraint, dict):
                        # Dictionary but maybe missing 'value' - try to find suitable content
                        constraint_value = ""
                        # Prioritize common keys if 'value' is missing
                        found_val = False
                        for potential_key in ['value', 'content', 'text', 'data']:
                             if potential_key in safe_constraint:
                                  constraint_value = str(safe_constraint[potential_key])
                                  found_val = True
                                  break
                        if not found_val:
                             # Fallback: concatenate string representations of values
                             constraint_value = "; ".join([f"{k}={str(v)}" for k, v in safe_constraint.items() if k not in ['key', 'judge']])

                        judge_info = safe_constraint.get("judge", {"method": "direct_gpt"})
                        if not isinstance(judge_info, dict):
                             judge_info = {"method": "direct_gpt"}

                        formatted_constraints.append({
                            "key": str(safe_constraint.get("key", f"constraint_{i+1}")),
                            "value": constraint_value,
                            "judge": {"method": str(judge_info.get("method", "direct_gpt"))}
                        })
                    else:
                        # Anything else - convert to string and wrap
                        formatted_constraints.append({
                            "key": f"constraint_{i+1}",
                            "value": str(safe_constraint),
                            "judge": {"method": "direct_gpt"}
                        })
                processed_constraints = formatted_constraints
        
        # --- Construct dictionaries for schema ---
        # Extra info dict with any fields we want to preserve
        extra_info = {
            "original_question": question or "",
            "aux_info": row.get("aux_info", "") or "",
            "infer_type": infer_type or "main",
            "split": row.get("split", "") or ""
        }
        
        # Construct reward model info
        reward_model = {}
        
        if tag == "P-Level":
            # For P-Level, include the answer in the reward model
            # Convert all answers to strings to match the schema
            if processed_answer:
                reward_model["answer"] = [str(item) for item in processed_answer]
            else:
                reward_model["answer"] = []
        elif tag == "C-Level":
            # For C-Level, include the constraints in the reward model
            # Ensure constraints have the expected structure for the schema
            if processed_constraints:
                formatted_constraints = []
                for constraint in processed_constraints:
                    # Make sure each constraint has the expected keys
                    if isinstance(constraint, dict):
                        formatted_constraint = {
                            "key": str(constraint.get("key", "")),
                            "value": str(constraint.get("value", "")),
                            "judge": {
                                "method": str(constraint.get("judge", {}).get("method", ""))
                            }
                        }
                        formatted_constraints.append(formatted_constraint)
                reward_model["constraints"] = formatted_constraints
            else:
                reward_model["constraints"] = []
                
        # --- Build final item ---
        processed_item = {
            "id": item_id,
            "tag": tag,
            # Store the PIL Image object directly in the list
            "images": [pil_image],
            "data_source": "mm_ifeval", # Will be overwritten with cli arg value
            "prompt": [ # Format prompt as a list of dicts
                {
                    "role": "user",
                    "content": f"<image>\n{question}"
                }
            ],
            "ability": "mmifeval",  # Add ability
            "extra_info": extra_info,  # Keep as dict
            "reward_model": reward_model,  # Keep as dict
            "ground_truth": "" # Will be filled below
        }
        
        # Ensure all parts of extra_info are strings
        processed_item["extra_info"] = {k: str(v) for k, v in processed_item["extra_info"].items()}

        # Convert reward model values to strings before creating ground_truth
        safe_reward_model = convert_numpy_numbers(reward_model)

        # Add ground_truth field containing all necessary data for evaluation
        if tag == "P-Level":
            # For P-Level, ground_truth is the expected answer list
            gt_data = {
                "type": "P-Level",
                "answer": safe_reward_model.get("answer", []),
                "question": question or "",
                # Include the original base64 string in ground_truth, not the PIL object
                "images": [{'image': image, 'path': None}]
            }
            try:
                processed_item["ground_truth"] = safe_json_dumps(gt_data)
            except Exception as e:
                logger.warning(f"Error serializing P-Level ground_truth for {item_id}: {e}")
                # Fallback to a simpler serialization
                processed_item["ground_truth"] = json.dumps({
                    "type": "P-Level",
                    "answer": [str(a) for a in safe_reward_model.get("answer", [])],
                    "question": question or ""
                })
        else:  # C-Level
            # For C-Level, ground_truth contains constraints with judge metadata
            gt_data = {
                "type": "C-Level",
                "constraints": safe_reward_model.get("constraints", []),
                "question": question or "",
                # Include the original base64 string in ground_truth, not the PIL object
                "images": [{'image': image, 'path': None}]
            }
            try:
                processed_item["ground_truth"] = safe_json_dumps(gt_data)
            except Exception as e:
                logger.warning(f"Error serializing C-Level ground_truth for {item_id}: {e}")
                # Try to produce a valid JSON string even if there are complex objects
                simplified_constraints = []
                for c in safe_reward_model.get("constraints", []):
                    if isinstance(c, dict):
                        # Ensure the constraint is safely serializable
                        simplified_constraint = {
                            "key": str(c.get("key", "")),
                            "value": str(c.get("value", "")),
                            "judge": {"method": "direct_gpt"}
                        }
                        simplified_constraints.append(simplified_constraint)
                
                processed_item["ground_truth"] = json.dumps({
                    "type": "C-Level",
                    "constraints": simplified_constraints,
                    "question": question or ""
                })
        
        return processed_item, None
        
    except Exception as e:
        # Catch-all for any unexpected errors during processing
        return None, {
            "id": row.get("id", f"unknown_{index}"),
            "reason": "processing_error",
            "details": f"Unexpected error: {str(e)}",
            "index": index,
            "recoverable": False,
            "error": str(e)
        }

# Helper function to recursively convert numpy numbers
def convert_numpy_numbers(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        # Convert array elements as well
        return [convert_numpy_numbers(item) for item in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: convert_numpy_numbers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_numbers(item) for item in obj]
    else:
        return obj # Return object itself if not a numpy type, list, or dict

def main():
    parser = argparse.ArgumentParser(description="Process MM-IFEval dataset into SCHEMA format and split into train/val sets.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the MM-IFEval TSV file.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for parquet files.")
    parser.add_argument("--val_percent", type=float, default=0.1, help="Percentage of data to use for validation (0-1).")
    parser.add_argument("--data_source_name", type=str, default="mm_ifeval", help="Name to use for the data source field.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 2, help="Number of worker processes for parallel processing.")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of items to process (for testing).")
    parser.add_argument("--keep_invalid_images", action="store_true", help="Keep items with invalid images (they'll be retained with placeholders).")
    parser.add_argument("--report_file", type=str, default="mmifeval_preprocessing_report.json", help="Path to save preprocessing report.")
    parser.add_argument("--log_file", type=str, default="mmifeval_preprocessing.log", help="Path to save log file.")

    args = parser.parse_args()
    
    # Set up logging to both console and file
    setup_file_logging(args.log_file)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    train_parquet_path = os.path.join(args.output_dir, "train.parquet")
    val_parquet_path = os.path.join(args.output_dir, "val.parquet")
    
    # Initialize report dictionary
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": args.data_path,
        "output_dir": args.output_dir,
        "validation_percent": args.val_percent,
        "data_source_name": args.data_source_name,
        "workers": args.num_workers,
        "max_items": args.max_items,
        "keep_invalid_images": args.keep_invalid_images,
        "statistics": {},
        "filtered_items": []
    }

    # Load data
    logger.info(f"Loading data from {args.data_path}...")
    try:
        # Adjust separator if needed (e.g., sep='\t')
        df = pd.read_csv(args.data_path, sep='\t', quoting=3) # quoting=3 helps with potential quote issues in TSV
        logger.info(f"Loaded {len(df)} rows.")
        report["statistics"]["total_rows"] = len(df)
    except Exception as e:
        logger.error(f"Failed to load data from {args.data_path}: {e}")
        report["error"] = f"Failed to load data: {str(e)}"
        with open(args.report_file, 'w') as f:
            json.dump(report, f, indent=2)
        return

    if args.max_items:
        logger.info(f"Processing a maximum of {args.max_items} items.")
        df = df.head(args.max_items)
        report["statistics"]["limited_to"] = args.max_items

    # --- Process Data in Parallel ---
    processed_examples = []
    filtered_items = []
    
    # Convert DataFrame rows to dictionaries to avoid pickling issues with pandas objects
    items_to_process = df.to_dict('records')

    logger.info(f"Starting parallel processing with {args.num_workers} workers...")
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(executor.map(process_item, enumerate(items_to_process)), total=len(items_to_process), desc="Processing items"))
    processing_time = time.time() - start_time
    report["statistics"]["processing_time_seconds"] = processing_time
    
    # Separate processed items and filtered items
    for result in results:
        processed_item, filter_info = result
        if processed_item:
            processed_examples.append(processed_item)
        elif filter_info:
            # If invalid_image but keep_invalid_images is true, try to recover
            if filter_info["reason"] == "invalid_image" and args.keep_invalid_images:
                item_index = filter_info["index"]
                logger.info(f"Recovering item with invalid image: {filter_info['id']}")
                
                # Create a minimal valid item
                row = items_to_process[item_index]
                try:
                    # Create a minimal valid item
                    recovered_item = {
                        "id": row.get("id", f"recovered_{item_index}"),
                        "tag": row.get("tag"),
                        # Store None for image in recovery path, or a placeholder PIL image if needed
                        "images": [None], # Or create a small blank PIL Image
                        "data_source": "mm_ifeval",
                        "prompt": [ # Format prompt as a list of dicts
                            {
                                "role": "user",
                                "content": f"<image>\n{row.get('question', '')}"
                            }
                        ],
                        "ability": "mmifeval",
                        "extra_info": {
                            "original_question": row.get("question", "") or "",
                            "aux_info": row.get("aux_info", "") or "",
                            "infer_type": row.get("infer_type", "main") or "main",
                            "split": row.get("split", "") or ""
                        },
                        "reward_model": {},
                        "ground_truth": "" # Will be filled based on tag
                    }
                    
                    # Add appropriate reward model based on tag
                    if row.get("tag") == "P-Level" and row.get("answer"):
                        try:
                            answer = row.get("answer")
                            if isinstance(answer, str):
                                answer = json.loads(answer)
                            # Convert to list of strings
                            if isinstance(answer, list):
                                recovered_item["reward_model"] = {"answer": [str(item) for item in answer]}
                                # Also add to ground_truth
                                recovered_item["ground_truth"] = json.dumps({
                                    "type": "P-Level",
                                    "answer": [str(item) for item in answer],
                                    "question": row.get("question", "") or ""
                                })
                            else:
                                recovered_item["reward_model"] = {"answer": [str(answer)]}
                                recovered_item["ground_truth"] = json.dumps({
                                    "type": "P-Level",
                                    "answer": [str(answer)],
                                    "question": row.get("question", "") or ""
                                })
                        except:
                            recovered_item["reward_model"] = {"answer": []}
                            recovered_item["ground_truth"] = json.dumps({
                                "type": "P-Level",
                                "answer": [],
                                "question": row.get("question", "") or ""
                            })
                            
                    elif row.get("tag") == "C-Level" and row.get("constraints"):
                        try:
                            constraints = row.get("constraints")
                            if isinstance(constraints, str):
                                constraints = json.loads(constraints)
                                
                            # Format constraints properly
                            formatted_constraints = []
                            if isinstance(constraints, list):
                                for constraint in constraints:
                                    if isinstance(constraint, dict):
                                        formatted_constraint = {
                                            "key": str(constraint.get("key", "")),
                                            "value": str(constraint.get("value", "")),
                                            "judge": {
                                                "method": str(constraint.get("judge", {}).get("method", ""))
                                            }
                                        }
                                        formatted_constraints.append(formatted_constraint)
                            
                            recovered_item["reward_model"] = {"constraints": formatted_constraints}
                            recovered_item["ground_truth"] = json.dumps({
                                "type": "C-Level",
                                "constraints": formatted_constraints,
                                "question": row.get("question", "") or ""
                            })
                        except:
                            recovered_item["reward_model"] = {"constraints": []}
                            recovered_item["ground_truth"] = json.dumps({
                                "type": "C-Level",
                                "constraints": [],
                                "question": row.get("question", "") or ""
                            })
                    
                    processed_examples.append(recovered_item)
                    filter_info["recovered"] = True
                except Exception as e:
                    logger.warning(f"Failed to recover item {filter_info['id']}: {e}")
                    filter_info["recovery_attempted"] = True
                    filter_info["recovery_error"] = str(e)
                
            filtered_items.append(filter_info)

    # Log statistics about filtering
    total_items = len(df)
    valid_items = len(processed_examples)
    filtered_count = len(filtered_items)
    
    # Count by reason
    filter_reasons = {}
    for item in filtered_items:
        reason = item.get("reason", "unknown")
        filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
    
    # Log detailed statistics
    logger.info(f"Started with {total_items} raw items")
    logger.info(f"Successfully processed {valid_items} valid items ({valid_items/total_items*100:.1f}%)")
    logger.info(f"Filtered out {filtered_count} items ({filtered_count/total_items*100:.1f}%)")
    
    # Log reasons for filtering
    logger.info("Filtering reasons:")
    for reason, count in filter_reasons.items():
        logger.info(f"  - {reason}: {count} items ({count/total_items*100:.1f}%)")
    
    # Update report
    report["statistics"].update({
        "total_items": total_items,
        "valid_items": valid_items,
        "filtered_items": filtered_count,
        "filtering_reasons": filter_reasons
    })
    
    # Add filtered items to report
    report["filtered_items"] = filtered_items

    if not processed_examples:
        logger.error("No valid examples were processed. Exiting.")
        report["error"] = "No valid examples processed"
        with open(args.report_file, 'w') as f:
            json.dump(report, f, indent=2)
        return

    # Assign data source name
    for example in processed_examples:
        example["data_source"] = args.data_source_name

    # --- Split Data ---
    random.seed(42) # for reproducibility
    random.shuffle(processed_examples)

    if args.val_percent > 0:
        split_index = int(len(processed_examples) * (1 - args.val_percent))
        train_data = processed_examples[:split_index]
        val_data = processed_examples[split_index:]
        logger.info(f"Split data into {len(train_data)} training and {len(val_data)} validation examples.")
        report["statistics"]["train_examples"] = len(train_data)
        report["statistics"]["validation_examples"] = len(val_data)
    else:
        train_data = processed_examples
        val_data = []
        logger.info(f"Using all {len(train_data)} examples for training (validation split percentage is 0).")
        report["statistics"]["train_examples"] = len(train_data)
        report["statistics"]["validation_examples"] = 0

    # --- Create and Save Datasets ---
    try:
        if train_data:
            train_dataset = datasets.Dataset.from_list(train_data, features=SCHEMA)
            logger.info(f"Saving training dataset to {train_parquet_path}...")
            train_dataset.to_parquet(train_parquet_path)
            report["train_parquet_path"] = train_parquet_path

        if val_data:
            val_dataset = datasets.Dataset.from_list(val_data, features=SCHEMA)
            logger.info(f"Saving validation dataset to {val_parquet_path}...")
            val_dataset.to_parquet(val_parquet_path)
            report["val_parquet_path"] = val_parquet_path

        logger.info("Datasets saved successfully!")
        
        # Save report
        with open(args.report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Preprocessing report saved to {args.report_file}")
        
        # Print summary with info about how to retrieve filtered items
        print("\n----- MM-IFEval Preprocessing Summary -----")
        print(f"Total rows processed: {total_items}")
        print(f"Valid items retained: {valid_items} ({valid_items/total_items*100:.1f}%)")
        print(f"Items filtered out: {filtered_count} ({filtered_count/total_items*100:.1f}%)")
        print(f"Data saved to: {args.output_dir}")
        print(f"Detailed report saved to: {args.report_file}")
        print(f"Complete logs saved to: {args.log_file}")
        print("\nTop filtering reasons:")
        top_reasons = sorted(filter_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
        for reason, count in top_reasons:
            print(f"  - {reason}: {count} items ({count/total_items*100:.1f}%)")
        print("\nTo recover filtered items, examine the report file and consider:")
        print("  - Using --keep_invalid_images to retain items with bad images")
        print("  - Manually fixing specific items using the item IDs from the report")
        print("--------------------------------------")

    except Exception as e:
        logger.error(f"Error creating or saving datasets: {e}", exc_info=True)
        report["error"] = f"Error creating or saving datasets: {str(e)}"
        with open(args.report_file, 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main() 