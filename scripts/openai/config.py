"""
Configuration settings for NLP-EHR project.

This module contains API keys, model identifiers, and other configuration
constants used throughout the project.
"""

# OpenAI API Configuration
OPENAI_API_KEY = "your_openai_api_key_here"  # Replace with your actual API key
OPENAI_FINE_TUNE_KEY = "your_openai_fine_tune_key_here"  # Replace with your actual fine-tune key

# OpenRouter API Configuration
OPENROUTER_API_KEY = "your_openrouter_api_key_here"  # Replace with your actual OpenRouter API key
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model Identifiers
GPT_MODELS = {
    'GPT4_O': 'gpt-4o-2024-08-06',
    'GPT4_MINI': 'gpt-4o-mini-2024-07-18',
    'GPT4_MINI_FINETUNED': 'replace-with-your-finetuned-model-id',  # Replace with your actual fine-tuned model ID
    'GEMMA_12B': 'google/gemma-3-12b-it'  # Add Gemma 3 via OpenRouter
}

# Default Configuration
DEFAULT_MODEL = GPT_MODELS['GPT4_MINI']
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
    GPT_MODELS['GPT4_O']: {
        'max_tokens': 8192,
        'temperature': DEFAULT_TEMPERATURE,
        'seed': DEFAULT_SEED,
        'provider': 'openai'
    },
    GPT_MODELS['GPT4_MINI']: {
        'max_tokens': 8192,
        'temperature': DEFAULT_TEMPERATURE,
        'seed': DEFAULT_SEED,
        'provider': 'openai'
    },
    GPT_MODELS['GPT4_MINI_FINETUNED']: {
        'max_tokens': 8192,
        'temperature': DEFAULT_TEMPERATURE,
        'seed': DEFAULT_SEED,
        'provider': 'openai'
    },
    GPT_MODELS['GEMMA_12B']: {
        #'max_tokens': 8192,
        'temperature': DEFAULT_TEMPERATURE,
        'seed': DEFAULT_SEED,
        'provider': 'openrouter'
    }
}
