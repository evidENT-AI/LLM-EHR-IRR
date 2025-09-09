#!/usr/bin/env python3

"""
Single letter analysis script using ChatGPT.

This script processes a single clinical letter using the ChatGPT model
for annotation and analysis.

Author: Liam Barrett
Version: 1.0.0
"""

import json

from labeller import ChatGPTInstance
from config import (
    OPENAI_API_KEY as DEFAULT_API_KEY,
    OPENROUTER_API_KEY,
    GPT_MODELS
)

import openai

def run_gemma_single():
    """Run single letter analysis with Gemma 3 12B via OpenRouter."""
    # Initialize with Gemma 3 model
    gpt = ChatGPTInstance(
        api_key=OPENROUTER_API_KEY,  # Use OpenRouter key
        model=GPT_MODELS['GEMMA_12B'],
        temperature=0,
        output_dir="../../results/annotations/llm/google/gemma3/",
        letter_id="letter_0000"  # Test with first letter
    )

    # Run analysis
    gpt.reset(letter_id="letter_0000")
    try:
        # load letter
        PREPROCESSED_LETTER = gpt.load_letter('../../data/mt_samples/letters/letter_0000.json')
        # run full annotation
        run_data = gpt.full_letter_annotation(PREPROCESSED_LETTER)
        print("Gemma 3 12B annotation completed successfully!")
        return run_data
    except Exception as e:
        print(f"Error during Gemma processing: {str(e)}")
        raise

def run_gpt_single():
    """Run single letter analysis with GPT-4o-mini."""
    # Initialize with output management
    gpt = ChatGPTInstance(
        api_key=DEFAULT_API_KEY,
        model=GPT_MODELS['GPT4_MINI'],
        temperature=0,
        output_dir="../../results/annotations/pilot_cot/openai/4o-mini/",
        letter_id="letter_0081"
    )

    # Run analysis
    gpt.reset(letter_id="letter_0081")
    try:
        # load letter
        PREPROCESSED_LETTER = gpt.load_letter('../../data/mt_samples/letters/letter_0081.json')
        # run full annotation
        run_data = gpt.full_letter_annotation(PREPROCESSED_LETTER)
        return run_data
    except Exception as e:
        print(f"Error during GPT processing: {str(e)}")
        raise

# Choose which to run
if __name__ == "__main__":
    # run_gpt_single()    # For GPT-4o-mini
    run_gemma_single()    # For Gemma 3 12B
