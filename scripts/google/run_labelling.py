#!/usr/bin/env python3

"""
Single letter analysis script using Google Gemini.

This script processes a single clinical letter using the Gemini model
for annotation and analysis.

Author: Liam Barrett (adapted for Google Gemini)
Version: 1.0.0
"""

import json

from labeller import GeminiInstance
from config import (
    GOOGLE_API_KEY as DEFAULT_API_KEY,
    GOOGLE_MODELS
)

# Initialize with output management
gemini = GeminiInstance(
    api_key=DEFAULT_API_KEY,
    model=GOOGLE_MODELS['GEMINI_PRO'],
    temperature=0,
    output_dir="../../results/annotations/llm/google/gemini-pro-1-5/",
    letter_id="letter_0058"
)

# Run analysis
gemini.reset(letter_id="letter_0058")
try:
    # load letter
    PREPROCESSED_LETTER = gemini.load_letter('../../data/mt_samples/letters/letter_0058.json')
    # run full annotation
    run_data = gemini.full_letter_annotation(PREPROCESSED_LETTER)
except FileNotFoundError as e:
    print(f"Letter file not found: {str(e)}")
except json.JSONDecodeError as e:
    print(f"Error parsing letter JSON: {str(e)}")
except ValueError as e:
    print(f"Value error in processing: {str(e)}")
except IOError as e:
    print(f"IO operation failed: {str(e)}")
except KeyError as e:
    print(f"Key error in data processing: {str(e)}")
except Exception as e:  # Catch any unexpected errors
    print(f"Unexpected error occurred: {str(e)}")
    raise  # Re-raise unexpected exceptions for debugging