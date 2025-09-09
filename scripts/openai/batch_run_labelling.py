#!/usr/bin/env python3

"""
Batch Letter Processing Module

This module provides functionality for batch processing clinical letters using
Large Language Models (LLMs). It includes a BatchLetterProcessor class that
handles the orchestration of processing multiple clinical letters across
different LLM models.

The module implements:
    - Batch processing of multiple clinical letters
    - Support for multiple LLM models
    - Comprehensive logging
    - Progress tracking with tqdm
    - Organized output management

Classes:
    BatchLetterProcessor: Main class for managing batch processing operations

Example:
    ```python
    from config import DEFAULT_API_KEY, GPT_MODELS

    # Initialize processor
    processor = BatchLetterProcessor(
        openai_key=DEFAULT_API_KEY,
        base_output_dir="path/to/output"
    )

    # Run batch processing
    models = [GPT_MODELS['GPT4_MINI']]
    processor.run_batch_processing(models)
    ```

Dependencies:
    - pathlib: For cross-platform path handling
    - logging: For comprehensive logging
    - datetime: For timestamp generation
    - typing: For type hints
    - tqdm: For progress tracking
    - labeller: Custom module for LLM interaction

Note:
    This script requires appropriate configuration in config.py and
    a valid OpenAI API key for operation.

Author: Liam Barrett
Version: 1.0.0
"""

from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

from labeller import ChatGPTInstance
from config import (
    OPENAI_API_KEY as DEFAULT_API_KEY,
    OPENAI_FINE_TUNE_KEY as FINE_TUNE_KEY,
    GPT_MODELS,
    LETTERS_DIR,
    EXAMPLE_LETTER_PATH,
    RESULTS_DIR
)

class BatchLetterProcessor:
    """Processes multiple clinical letters using specified LLM models."""
    def __init__(
        self,
        openai_key: str,
        base_output_dir: str,
        example_letter_path: str = EXAMPLE_LETTER_PATH,
        letters_dir: str = LETTERS_DIR
    ):
        """
        Initialize the BatchLetterProcessor.

        Args:
            openai_key (str): OpenAI API key for the LLM service
            base_output_dir (str): Base directory for output files
            example_letter_path (str, optional): Path to example letter.
                Defaults to EXAMPLE_LETTER_PATH from config
            letters_dir (str, optional): Directory containing letters to process.
                Defaults to LETTERS_DIR from config
        """
        self.api_key = openai_key
        self.base_output_dir = Path(base_output_dir)
        self.example_letter_path = Path(example_letter_path)
        self.letters_dir = Path(letters_dir)
        self.setup_logging()
    def setup_logging(self):
        """Configure logging for batch processing."""
        log_dir = self.base_output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"batch_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("BatchProcessor")
    def get_letter_files(self, start_index: int = 0) -> List[Path]:
        """
        Get list of all letter files to process, starting from a specified index.

        Args:
            start_index: The starting index (e.g., 10 for letter_0010)
                        Letters with indices lower than this will be skipped

        Returns:
            List of Path objects for the letters to process
        """
        # Get all letter files
        all_letter_files = sorted(self.letters_dir.glob("letter_*.json"))

        # Filter to only include letters with indices >= start_index
        filtered_files = []
        for file_path in all_letter_files:
            # Extract the index number from the filename
            filename = file_path.stem  # Gets filename without extension
            # Extract the numeric part (assumes format "letter_XXXX")
            try:
                index = int(filename.split('_')[1])
                if index >= start_index:
                    filtered_files.append(file_path)
            except (IndexError, ValueError):
                # Skip files that don't match the expected format
                continue

        return filtered_files
    def process_letter(self, letter_path: Path, model: str, output_dir: Path) -> Dict:
        """Process a single letter with specified model."""
        letter_id = letter_path.stem
        self.logger.info("Processing %s with model %s", letter_id, model)
        try:
            # Initialize model instance
            gpt = ChatGPTInstance(
                api_key=self.api_key,
                model=model,
                temperature=0,
                output_dir=str(output_dir),
                letter_id=letter_id,
                example_letter_path=str(self.example_letter_path)
            )
            # Reset for fresh start
            gpt.reset(letter_id=letter_id)
            # Process letter
            preprocessed_letter = gpt.load_letter(str(letter_path))
            _ = gpt.full_letter_annotation(preprocessed_letter)
            self.logger.info("Successfully processed %s", letter_id)
            return {"status": "success", "letter_id": letter_id}
        except Exception as e:
            self.logger.error("Failed to process %s: %s", letter_id, str(e))
            return {"status": "error", "letter_id": letter_id, "error": str(e)}
    def run_batch_processing(self, models: List[str], start_index: int = 0):
        """Run batch processing for all letters with specified models."""
        letters = self.get_letter_files(start_index)
        self.logger.info(
            "Starting batch processing of %d letters with %d models",
            len(letters),
            len(models)
        )
        for model in models:
            model_output_dir = self.base_output_dir / model.replace(".", "-")
            model_output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("Processing with model: %s", model)
            for letter_path in tqdm(letters, desc=f"Processing letters with {model}"):
                self.process_letter(letter_path, model, model_output_dir)

def run_gemma_batch():
    """Run batch processing with Gemma 3 12B via OpenRouter."""
    from config import OPENROUTER_API_KEY

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = f"{RESULTS_DIR}/annotations/llm/google/gemma3/{timestamp}/"

    # Use Gemma 3 12B model
    models = [GPT_MODELS['GEMMA_12B']]

    processor = BatchLetterProcessor(
        openai_key=OPENROUTER_API_KEY,  # Use OpenRouter key
        base_output_dir=base_output_dir
    )

    processor.run_batch_processing(models, start_index=16)
    print(f"\nGemma batch processing complete! Results saved to: {base_output_dir}")
    return base_output_dir

def main():
    """Run batch processing of clinical letters using specified models."""
    # Configuration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = f"{RESULTS_DIR}/annotations/march2025/batch_run_{timestamp}/"
    # Models to test
    models = [GPT_MODELS['GPT4_MINI_FINETUNED']]
    # Initialize and run batch processor
    processor = BatchLetterProcessor(
        openai_key=FINE_TUNE_KEY,
        base_output_dir=base_output_dir
    )
    processor.run_batch_processing(models, start_index=80)
    print("\nBatch processing complete!")

if __name__ == "__main__":
    # Choose which to run:
    main()              # For regular OpenAI models
    # run_gemma_batch()     # For Gemma 3
