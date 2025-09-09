"""
Configuration settings for NLP-EHR project.

This module contains API keys, model identifiers, and other configuration
constants used throughout the project.
"""

# Anthropic API Configuration
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"  # Replace with your actual API key

# Model Identifiers
CLAUDE_MODELS = {
    'CLAUDE_OPUS': 'claude-3-opus-20240229',
    'CLAUDE_SONNET_35': 'claude-3-5-sonnet-20241022',
    'CLAUDE_SONNET_37': 'claude-3-7-sonnet-20250219',
    'CLAUDE_HAIKU': 'claude-3-5-haiku-20241022'
}

# Default Configuration
DEFAULT_MODEL = CLAUDE_MODELS['CLAUDE_HAIKU']
DEFAULT_TEMPERATURE = 0
DEFAULT_SYSTEM_SEED = 42

# Path Configuration
DATA_DIR = "../../data"
MT_SAMPLES_DIR = f"{DATA_DIR}/mt_samples"
LETTERS_DIR = f"{MT_SAMPLES_DIR}/letters"
EXAMPLE_LETTER = "letter_0080.json"
EXAMPLE_LETTER_PATH = f"{LETTERS_DIR}/{EXAMPLE_LETTER}"
RESULTS_DIR = "../../results"

# Model Parameters
MODEL_PARAMS = {
    CLAUDE_MODELS['CLAUDE_OPUS']: {
        'max_tokens': 4096,
        'temperature': DEFAULT_TEMPERATURE,
        'system_seed': DEFAULT_SYSTEM_SEED
    },
    CLAUDE_MODELS['CLAUDE_SONNET_35']: {
        'max_tokens': 8192,
        'temperature': DEFAULT_TEMPERATURE,
        'system_seed': DEFAULT_SYSTEM_SEED
    },
    CLAUDE_MODELS['CLAUDE_SONNET_37']: {
        'max_tokens': 8192,
        'temperature': DEFAULT_TEMPERATURE,
        'system_seed': DEFAULT_SYSTEM_SEED
    },
    CLAUDE_MODELS['CLAUDE_HAIKU']: {
        'max_tokens': 8192,
        'temperature': DEFAULT_TEMPERATURE,
        'system_seed': DEFAULT_SYSTEM_SEED
    }
}
