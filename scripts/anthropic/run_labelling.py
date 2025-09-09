#!/usr/bin/env python3

"""
Single letter analysis script using Claude.

This script processes a single clinical letter using the Claude model
for annotation and analysis.

Author: Liam Barrett
Version: 1.0.0
"""

import json

from labeller import ClaudeInstance  # Update to new class name
from config import (
    ANTHROPIC_API_KEY as DEFAULT_API_KEY,
    CLAUDE_MODELS
)

from anthropic import APIError

# Initialize with output management
gpt = ClaudeInstance(
    api_key=DEFAULT_API_KEY,
    model=CLAUDE_MODELS['CLAUDE_SONNET_35'],
    temperature=0,
    output_dir="../../results/annotations/llm/anthropic/claude-sonnet-3-5",
    letter_id="letter_0062"
)

# Run analysis
gpt.reset(letter_id="letter_0062")
try:
    # load letter
    PREPROCESSED_LETTER = gpt.load_letter('../../data/mt_samples/letters/letter_0062.json')
    # run full annotation
    run_data = gpt.full_letter_annotation(PREPROCESSED_LETTER)
except FileNotFoundError as e:
    print(f"Letter file not found: {str(e)}")
except json.JSONDecodeError as e:
    print(f"Error parsing letter JSON: {str(e)}")
except APIError as e:  # Update to Anthropic errors
    print(f"Anthropic API error: {str(e)}")
except ValueError as e:
    print(f"Value error in processing: {str(e)}")
except IOError as e:
    print(f"IO operation failed: {str(e)}")
except KeyError as e:
    print(f"Key error in data processing: {str(e)}")
except Exception as e:  # Catch any unexpected errors
    print(f"Unexpected error occurred: {str(e)}")
    raise  # Re-raise unexpected exceptions for debugging
