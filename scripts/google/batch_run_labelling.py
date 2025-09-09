#!/usr/bin/env python3

"""
Batch Letter Processing Module for Google Gemini

This module provides functionality for batch processing clinical letters using
Google's Gemini models, with built-in handling for API rate limits and quotas.

The module implements:
    - Batch processing of multiple clinical letters
    - Support for multiple Gemini models
    - Comprehensive logging
    - Progress tracking with tqdm
    - Organized output management
    - API rate limit and quota exhaustion handling

Classes:
    BatchLetterProcessor: Main class for managing batch processing operations

Example:
    ```python
    from config import GOOGLE_API_KEY, GEMINI_MODELS

    # Initialize processor
    processor = BatchLetterProcessor(
        google_key=GOOGLE_API_KEY,
        base_output_dir="path/to/output"
    )

    # Run batch processing
    models = [GEMINI_MODELS['GEMINI_PRO']]
    processor.run_batch_processing(models)
    ```

Author: Liam Barrett
Version: 1.0.0
"""

from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
import time
import requests
import sys

from labeller import GeminiInstance
from config import (
    GOOGLE_API_KEY as DEFAULT_API_KEY,
    GOOGLE_MODELS,
    LETTERS_DIR,
    EXAMPLE_LETTER_PATH,
    RESULTS_DIR
)

# Error classes for Google API issues
class GoogleAPIError(Exception):
    """Base exception for Google API errors"""
    pass

class QuotaExceededError(GoogleAPIError):
    """Exception for when the Google API quota is exceeded"""
    pass

class RateLimitError(GoogleAPIError):
    """Exception for when the Google API rate limit is reached"""
    pass

class BatchLetterProcessor:
    """Processes multiple clinical letters using specified Gemini models."""
    def __init__(
        self,
        google_key: str,
        base_output_dir: str,
        example_letter_path: str = EXAMPLE_LETTER_PATH,
        letters_dir: str = LETTERS_DIR,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        """
        Initialize the BatchLetterProcessor.

        Args:
            google_key (str): Google API key for the Gemini service
            base_output_dir (str): Base directory for output files
            example_letter_path (str, optional): Path to example letter.
                Defaults to EXAMPLE_LETTER_PATH from config
            letters_dir (str, optional): Directory containing letters to process.
                Defaults to LETTERS_DIR from config
            max_retries (int, optional): Maximum number of retries for API calls.
                Defaults to 3.
            retry_delay (int, optional): Delay in seconds between retries.
                Defaults to 5.
        """
        self.api_key = google_key
        self.base_output_dir = Path(base_output_dir)
        self.example_letter_path = Path(example_letter_path)
        self.letters_dir = Path(letters_dir)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.setup_logging()
        self.api_exhausted = False
        self.processed_letters = []

    def setup_logging(self):
        """Configure logging for batch processing."""
        log_dir = self.base_output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"batch_processing_{timestamp}.log"

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("BatchProcessor")

        # Also save processed letters info
        self.progress_file = log_dir / f"processing_progress_{timestamp}.txt"
        with open(self.progress_file, 'w') as f:
            f.write(f"Batch processing started at {timestamp}\n")
            f.write("Format: [letter_id] [model] [status]\n\n")

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
        if self.api_exhausted:
            return {"status": "skipped", "letter_id": letter_path.stem, "error": "API quota exhausted"}

        letter_id = letter_path.stem
        self.logger.info("Processing %s with model %s", letter_id, model)

        for attempt in range(self.max_retries):
            try:
                # Initialize model instance
                gemini = GeminiInstance(
                    api_key=self.api_key,
                    model=model,
                    temperature=0,
                    output_dir=str(output_dir),
                    letter_id=letter_id,
                    example_letter_path=str(self.example_letter_path)
                )

                # Reset for fresh start
                gemini.reset(letter_id=letter_id)

                # Process letter
                preprocessed_letter = gemini.load_letter(str(letter_path))
                _ = gemini.full_letter_annotation(preprocessed_letter)

                # Log success
                self.logger.info("Successfully processed %s", letter_id)
                self._update_progress(letter_id, model, "success")

                return {"status": "success", "letter_id": letter_id}

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    self.logger.warning(f"Rate limit hit for {letter_id}. Attempt {attempt+1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        self.logger.error("Rate limit persists, marking API as exhausted")
                        self.api_exhausted = True
                        self._update_progress(letter_id, model, f"failed: rate limit exceeded")
                        raise RateLimitError("Google API rate limit exceeded") from e

                elif e.response.status_code in (403, 401):  # Forbidden, Unauthorized (quota issues)
                    self.logger.error("API quota exceeded or authentication issue. Stopping batch processing.")
                    self.api_exhausted = True
                    self._update_progress(letter_id, model, f"failed: quota exceeded")
                    raise QuotaExceededError("Google API quota exceeded") from e

                else:
                    self.logger.warning(f"HTTP error for {letter_id}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        self._update_progress(letter_id, model, f"failed: {str(e)}")
                        return {"status": "error", "letter_id": letter_id, "error": str(e)}

            except Exception as e:
                # Check for specific Google API errors in the exception message
                error_msg = str(e).lower()
                if "quota" in error_msg or "limit exceeded" in error_msg:
                    self.logger.error("API quota exceeded. Stopping batch processing.")
                    self.api_exhausted = True
                    self._update_progress(letter_id, model, f"failed: quota exceeded")
                    raise QuotaExceededError("Google API quota exceeded") from e
                elif "rate limit" in error_msg:
                    self.logger.warning(f"Rate limit hit for {letter_id}. Attempt {attempt+1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        self.logger.error("Rate limit persists, marking API as exhausted")
                        self.api_exhausted = True
                        self._update_progress(letter_id, model, f"failed: rate limit exceeded")
                        raise RateLimitError("Google API rate limit exceeded") from e
                else:
                    self.logger.error(f"Failed to process {letter_id}: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        self._update_progress(letter_id, model, f"failed: {str(e)}")
                        return {"status": "error", "letter_id": letter_id, "error": str(e)}

        # This should not be reached due to the return statements in the loops,
        # but adding as a fallback
        return {"status": "error", "letter_id": letter_id, "error": "Unknown error"}

    def _update_progress(self, letter_id: str, model: str, status: str):
        """Update the progress file with latest processing status."""
        with open(self.progress_file, 'a') as f:
            f.write(f"{letter_id} {model} {status}\n")
        # Also store in memory
        self.processed_letters.append({"letter_id": letter_id, "model": model, "status": status})

    def run_specific_letters(self, models: List[str], letter_ids: List[str]):
        """Run batch processing for specific letters only."""
        # Convert letter_ids to actual file paths
        letters = []
        for letter_id in letter_ids:
            letter_path = self.letters_dir / f"{letter_id}.json"
            if letter_path.exists():
                letters.append(letter_path)
            else:
                self.logger.warning(f"Letter file not found: {letter_path}")

        total_letters = len(letters)
        total_models = len(models)
        self.logger.info(
            "Starting batch processing of %d specific letters with %d models",
            total_letters,
            total_models
        )

        try:
            for model in models:
                if self.api_exhausted:
                    self.logger.warning(f"Skipping model {model} due to API quota exhaustion")
                    continue

                model_output_dir = self.base_output_dir / model.replace("/", "-").replace(".", "-")
                model_output_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info("Processing with model: %s", model)

                # Create progress bar
                progress_bar = tqdm(total=len(letters), desc=f"Processing letters with {model}")

                for letter_path in letters:
                    if self.api_exhausted:
                        self.logger.warning(f"Skipping remaining letters due to API quota exhaustion")
                        break

                    try:
                        result = self.process_letter(letter_path, model, model_output_dir)
                        progress_bar.update(1)
                        if result["status"] != "success":
                            self.logger.warning(f"Letter {letter_path.stem} processing failed: {result.get('error', 'Unknown error')}")
                    except (QuotaExceededError, RateLimitError) as e:
                        self.logger.error(f"API limit reached: {str(e)}")
                        self.api_exhausted = True
                        break
                    except Exception as e:
                        self.logger.error(f"Unexpected error processing {letter_path.stem}: {str(e)}")
                        progress_bar.update(1)
                        continue

                progress_bar.close()

        except KeyboardInterrupt:
            self.logger.info("Batch processing interrupted by user")
        finally:
            # Save summary of processing results
            completed = len([l for l in self.processed_letters if l["status"] == "success"])
            self.logger.info(f"Batch processing summary: {completed}/{total_letters * total_models} letters processed successfully")
            if self.api_exhausted:
                self.logger.info("Processing stopped due to API quota exhaustion")

            # Write final summary to progress file
            with open(self.progress_file, 'a') as f:
                f.write(f"\nProcessing ended at {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
                f.write(f"Summary: {completed}/{total_letters * total_models} letters processed successfully\n")
                if self.api_exhausted:
                    f.write("Processing stopped due to API quota exhaustion\n")

            return {
                "processed": completed,
                "total": total_letters * total_models,
                "api_exhausted": self.api_exhausted,
                "processed_letters": self.processed_letters
            }

    def run_batch_processing(self, models: List[str]):
        """Run batch processing for all letters with specified models."""
        letters = self.get_letter_files(40)
        total_letters = len(letters)
        total_models = len(models)
        self.logger.info(
            "Starting batch processing of %d letters with %d models",
            total_letters,
            total_models
        )

        try:
            for model in models:
                if self.api_exhausted:
                    self.logger.warning(f"Skipping model {model} due to API quota exhaustion")
                    continue

                model_output_dir = self.base_output_dir / model.replace(".", "-")
                model_output_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info("Processing with model: %s", model)

                # Create progress bar that can be updated even if exception occurs
                progress_bar = tqdm(total=len(letters), desc=f"Processing letters with {model}")

                for letter_path in letters:
                    if self.api_exhausted:
                        self.logger.warning(f"Skipping remaining letters due to API quota exhaustion")
                        break

                    try:
                        result = self.process_letter(letter_path, model, model_output_dir)
                        progress_bar.update(1)
                        if result["status"] != "success":
                            self.logger.warning(f"Letter {letter_path.stem} processing failed: {result.get('error', 'Unknown error')}")
                    except (QuotaExceededError, RateLimitError) as e:
                        self.logger.error(f"API limit reached: {str(e)}")
                        self.api_exhausted = True
                        break
                    except Exception as e:
                        self.logger.error(f"Unexpected error processing {letter_path.stem}: {str(e)}")
                        progress_bar.update(1)
                        continue

                progress_bar.close()

        except KeyboardInterrupt:
            self.logger.info("Batch processing interrupted by user")
        finally:
            # Save summary of processing results
            completed = len([l for l in self.processed_letters if l["status"] == "success"])
            self.logger.info(f"Batch processing summary: {completed}/{total_letters * total_models} letters processed successfully")
            if self.api_exhausted:
                self.logger.info("Processing stopped due to API quota exhaustion")

            # Write final summary to progress file
            with open(self.progress_file, 'a') as f:
                f.write(f"\nProcessing ended at {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
                f.write(f"Summary: {completed}/{total_letters * total_models} letters processed successfully\n")
                if self.api_exhausted:
                    f.write("Processing stopped due to API quota exhaustion\n")

            return {
                "processed": completed,
                "total": total_letters * total_models,
                "api_exhausted": self.api_exhausted,
                "processed_letters": self.processed_letters
            }

def run_specifc_letters():
    """Run batch processing of clinical letters using specified models."""
    # Configuration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = f"{RESULTS_DIR}/annotations/llm/google/gemini-pro-1-5/"
    # Models to test
    models = [GOOGLE_MODELS['GEMINI_PRO']]
    # Initialize and run batch processor
    processor = BatchLetterProcessor(
      google_key=DEFAULT_API_KEY,
      base_output_dir=base_output_dir
    )

    # Specify only the missing letters
    missing_letters = ["letter_0018", "letter_0019", "letter_0020",
                       "letter_0030", "letter_0033", "letter_0036",
                       "letter_0041", "letter_0047", "letter_0055",
                       "letter_0058", "letter_0060", "letter_0065"]
    # Process only missing letters
    results = processor.run_specific_letters(models, missing_letters)
    # Print summary
    if results["api_exhausted"]:
        print("\nBatch processing stopped due to API quota exhaustion")
    print(f"\nBatch processing completed: {results['processed']}/{results['total']} letters processed successfully")
    print(f"Results saved to: {base_output_dir}")

def main():
    """Run batch processing of clinical letters using specified models."""
    # Configuration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = f"{RESULTS_DIR}/annotations/llm/google/gemini-pro-1-5/"

    # Models to test
    models = [GOOGLE_MODELS['GEMINI_PRO']]

    # Initialize and run batch processor
    processor = BatchLetterProcessor(
        google_key=DEFAULT_API_KEY,
        base_output_dir=base_output_dir
    )

    results = processor.run_batch_processing(models)

    # Print summary
    if results["api_exhausted"]:
        print("\nBatch processing stopped due to API quota exhaustion")
    print(f"\nBatch processing completed: {results['processed']}/{results['total']} letters processed successfully")
    print(f"Results saved to: {base_output_dir}")

if __name__ == "__main__":
    #main()
    run_specifc_letters()
