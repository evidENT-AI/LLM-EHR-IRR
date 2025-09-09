"""
Configuration settings for NLP-EHR project.

This module contains API keys, model identifiers, Google Cloud project settings,
and other configuration constants used throughout the project.
"""

# Google Cloud Configuration
PROJECT_ID = "your_gcp_project_id_here"  # Replace with your actual GCP project ID
LOCATION = "us-central1"  # LLAMA 3.1 is only available in us-central1

# LLAMA 3.1 Model Identifiers
LLAMA_MODELS = {
    'LLAMA_8B': 'meta/llama-3.1-8b-instruct-maas',
    'LLAMA_70B': 'meta/llama-3.1-70b-instruct-maas',
    'LLAMA_405B': 'meta/llama-3.1-405b-instruct-maas'
}

# Default Configuration
DEFAULT_MODEL = LLAMA_MODELS['LLAMA_405B']
DEFAULT_TEMPERATURE = 0
DEFAULT_SEED = 42
DEFAULT_APPLY_LLAMA_GUARD = True

# Path Configuration
DATA_DIR = "../../data"
MT_SAMPLES_DIR = f"{DATA_DIR}/mt_samples"
LETTERS_DIR = f"{MT_SAMPLES_DIR}/letters"
EXAMPLE_LETTER = "letter_0080.json"
EXAMPLE_LETTER_PATH = f"{LETTERS_DIR}/{EXAMPLE_LETTER}"
RESULTS_DIR = "../../results"

# Model Parameters
MODEL_PARAMS = {
    LLAMA_MODELS['LLAMA_8B']: {
        'max_tokens': 8192,
        'temperature': DEFAULT_TEMPERATURE,
        'seed': DEFAULT_SEED,
        'apply_llama_guard': DEFAULT_APPLY_LLAMA_GUARD
    },
    LLAMA_MODELS['LLAMA_70B']: {
        'max_tokens': 8192,
        'temperature': DEFAULT_TEMPERATURE,
        'seed': DEFAULT_SEED,
        'apply_llama_guard': DEFAULT_APPLY_LLAMA_GUARD
    },
    LLAMA_MODELS['LLAMA_405B']: {
        'max_tokens': 8192,
        'temperature': DEFAULT_TEMPERATURE,
        'seed': DEFAULT_SEED,
        'apply_llama_guard': DEFAULT_APPLY_LLAMA_GUARD
    }
}
