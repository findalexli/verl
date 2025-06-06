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

import re
import numpy as np
from typing import Dict, List, Optional, Union

# Removed unused imports: RewardField, RewardScore
from .math import last_boxed_only_string, remove_boxed


# Regex to extract numbers from text, moved outside the class
number_pattern = re.compile(r'\b(\d+)\b')

def extract_number(text: str) -> Optional[int]:
    """Extract the last number from text, prioritizing boxed answers, or None if no number is found."""
    if not text:
        return None
    
    # First check if there's a boxed answer
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        try:
            # Extract the content from the boxed expression
            boxed_content = remove_boxed(boxed)
            # Try to convert to integer
            return int(boxed_content.strip())
        except (ValueError, TypeError, AssertionError):
            # If conversion fails or boxed format is unexpected, fall back to regular number extraction
            # Try to extract numbers from the boxed string directly
            boxed_matches = number_pattern.findall(boxed)
            if boxed_matches:
                try:
                    return int(boxed_matches[-1])
                except (ValueError, TypeError):
                    pass
    
    # Find all numbers in the text
    matches = number_pattern.findall(text)
    if not matches:
        return None
    
    # Return the last number found
    return int(matches[-1])


def compute_score(solution_str: str, ground_truth: str) -> float:
    """
    Compute the repetition counting score for a given solution and ground truth.
    
    Args:
        solution_str: Model's output text
        ground_truth: The expected number of repetitions as a string
        
    Returns:
        Score between 0.0 and 1.0
    """
    # Extract rep count from model output
    predicted_count = extract_number(solution_str)
    
    # Parse ground truth count
    try:
        true_count = int(ground_truth.strip())
    except (ValueError, TypeError):
        # If ground truth is not a valid number, score is 0
        return 0.0
    
    # If model didn't produce a number, score is 0
    if predicted_count is None:
        return 0.0
    
    # Calculate absolute difference
    diff = abs(predicted_count - true_count)
    # Version 1 of reward score, using a linear percenrtgae 
    # Use a scoring function based on the difference
    # if diff == 0:
    #     score = 1.0  # Exactly correct
    # else:
    #     # Use relative error with a minimum denominator to avoid large swings for small counts
    #     denominator = max(true_count, 5) # Using 5 as a minimum denominator
    #     # Cap the relative error at 1.0 (so score doesn't go below 0.0)
    #     relative_error = min(diff / denominator, 1.0)
    #     # Score is 1 minus relative error
    #     score = max(0.0, 1.0 - relative_error)
    #     # Round to 2 decimal places for consistency
    #     score = round(score, 2)
    # Version 2 of reward score: needs to be within 1 at most off by 1
    # if diff > 1:
    #     return 0
    # else:
    #     return 1
    # Version 3: reward score need to be within 2 (at most differ by 2)
    # if diff > 2:
    #     return 0
    # else:
    #     return 1
    # version 4 exactly the same 
    if diff > 1:
        return 0
    else:
        return 1