"""
Configuration settings for NLP-EHR project.

This module contains API keys, model identifiers, and other configuration
constants used throughout the project.
"""

# Google API Configuration
GOOGLE_API_KEY = "your_google_api_key_here"  # Replace with your actual API key

# Model Identifiers
GOOGLE_MODELS = {
    'GEMINI_PRO': 'gemini-1.5-pro',
    'GEMINI_FLASH': 'gemini-1.5-flash',
    'GEMMA3_12B': 'gemma-3-12b-it'
}

# Default Configuration
DEFAULT_MODEL = GOOGLE_MODELS['GEMINI_PRO']
DEFAULT_TEMPERATURE = 0
DEFAULT_SEED = 42

# Path Configuration
DATA_DIR = "../../data"
MT_SAMPLES_DIR = f"{DATA_DIR}/mt_samples"
LETTERS_DIR = f"{MT_SAMPLES_DIR}/letters"
EXAMPLE_LETTER = "letter_0080.json"
EXAMPLE_LETTER_PATH = f"{LETTERS_DIR}/{EXAMPLE_LETTER}"
RESULTS_DIR = "../../results"

# Model Parameters
MODEL_PARAMS = {
    GOOGLE_MODELS['GEMINI_PRO']: {
        'max_tokens': 8192,
        'temperature': DEFAULT_TEMPERATURE,
        'seed': DEFAULT_SEED
    },
    GOOGLE_MODELS['GEMINI_FLASH']: {
        'max_tokens': 8192,
        'temperature': DEFAULT_TEMPERATURE,
        'seed': DEFAULT_SEED
    },
    GOOGLE_MODELS['GEMMA3_12B']: {
        'max_tokens': 8192,
        'temperature': DEFAULT_TEMPERATURE,
        'seed': DEFAULT_SEED
    }
}