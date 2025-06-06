# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
from typing import Any, Dict

import jsonschema


def strip_markdown_json(text: str) -> str:
    """Removes markdown code fences (```json ... ```) from a string."""
    text = (
        text.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    return text


def check_json_format(predict_str: str) -> float:
    """
    Checks if the prediction string is valid JSON.
    Handles potential markdown code blocks.
    Returns 1.0 if valid JSON, 0.0 otherwise.
    """
    cleaned_str = strip_markdown_json(predict_str)
    if not cleaned_str:  # Empty string is not valid JSON
        return 0.0
    try:
        json.loads(cleaned_str)
        return 1.0
    except ValueError:
        return 0.0


def check_schema_compliance(predict_str: str, schema: Dict[str, Any]) -> float:
    """
    Checks if the prediction string is valid JSON and conforms to the provided schema.
    Returns 1.0 if valid and compliant, 0.0 otherwise.
    """
    # First, check if it's even valid JSON
    if check_json_format(predict_str) == 0.0:
        return 0.0

    cleaned_str = strip_markdown_json(predict_str)
    try:
        instance = json.loads(cleaned_str)
        jsonschema.validate(instance=instance, schema=schema)
        return 1.0
    except (ValueError, jsonschema.ValidationError):
        # ValueError should theoretically be caught by check_json_format,
        # but included for robustness.
        return 0.0
    except Exception: # Catch potential other errors during validation
        return 0.0


def compute_score(predict_str: str, ground_truth: str, extra_info: Dict[str, Any]) -> float:
    """
    Computes a combined score based on JSON format and schema compliance.
    - 0.5 points for valid JSON format.
    - 0.5 points for schema compliance (only awarded if format is valid).
    Args:
        predict_str: The model's generated output string.
        ground_truth: The ground truth JSON string (not directly used for scoring here).
        extra_info: Dictionary containing 'schema_str', a string representation of the JSON schema.
    Returns:
        Combined score (0.0 to 1.0).
    """
    # First check if prediction is valid JSON
    json_format_score = check_json_format(predict_str)
    
    # If not valid JSON, return 0
    if json_format_score == 0.0:
        return 0.0

    # Check if schema string is provided
    schema_str = extra_info.get("schema_str") if extra_info else None
    if not schema_str:
        # If no schema is provided, only judge based on JSON validity
        return json_format_score
    
    # Parse the schema string into a schema object
    try:
        schema = json.loads(schema_str)
    except (json.JSONDecodeError, TypeError):
        # If schema string is not valid JSON, return only format score
        return json_format_score
    
    # Check schema compliance
    schema_compliance_score = check_schema_compliance(predict_str, schema)
    
    # Simple weighted score: 0.5 for format, 0.5 for compliance
    final_score = 0.5 * json_format_score + 0.5 * schema_compliance_score
    return final_score 