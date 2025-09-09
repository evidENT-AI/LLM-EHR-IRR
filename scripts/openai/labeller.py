#!/usr/bin/env python3

"""
Clinical Letter Analysis Module using ChatGPT

This module provides a ChatGPT-based implementation for analysing clinical letters,
with capabilities for preprocessing, annotation, and result management. It includes
functionality for in-context learning and comprehensive logging of the analysis process.

The main class, ChatGPTInstance, handles:
    - Clinical letter preprocessing and loading
    - ChatGPT interaction and prompt management
    - JSON response validation and parsing
    - Output management and logging
    - In-context learning using example letters
    - Full letter annotation pipeline

Example:
    ```python
    api_key = "your-api-key"
    with ChatGPTInstance(api_key) as gpt:
        letter = gpt.load_letter("path/to/letter.json")
        results = gpt.full_letter_annotation(letter)
    ```

Dependencies:
    - openai: For ChatGPT API interaction
    - pathlib: For cross-platform path handling
    - logging: For comprehensive logging
    - json: For data serialization
    - datetime: For timestamp generation
    - uuid: For unique identifier generation

The module requires a valid OpenAI API key and appropriate permissions
for file operations in the specified output directory.

Module requires snowstorm and elastic search servers to be running locally:
```bash
cd /opt/elasticsearch
./bin/elasticsearch

# go to where snowstorm jar is located
java -Xms2g -Xmx4g -jar snowstorm-10.5.1.jar --snowstorm.rest-api.readonly=true
```
Author: Liam Barrett
Version: 1.1.0
"""

from pathlib import Path
import logging
from datetime import datetime
import uuid
import json

from parser import LLMOutputParser
from prompts import (
    PREPROMPT,
    EXAMPLE_LETTER_PROMPT,
    create_letter_prompt,
    create_feedback_prompt,
    SNOMED_CODE_SELECTION_PROMPT,
    SNOMED_SEARCH_REFINEMENT_PROMPT
)
from snomed_searcher import SnomedSearcher

import openai
import pandas as pd

class ChatGPTInstance:
    """Instance of ChatGPT for clinical letter analysis with preprocessing capability."""

    def __init__(self, api_key, model="gpt-3.5-turbo", temperature=0, seed=42,
                 output_dir="outputs", letter_id=None,
                 example_letter_path='../../data/mt_samples/letters/letter_0080.json',
                 example_annotation_path='../../data/annotated_letter_0080.json'):
        # Initialize core attributes
        self.api_key = api_key
        self.model_name = model
        self.base_temperature = temperature
        self.base_seed = seed
        self.base_output_dir = Path(output_dir)
        self.example_letter_path = Path(example_letter_path)
        self.example_annotation_path = Path(example_annotation_path)
        # Initialize state
        self.initialize_state(letter_id)

    def initialize_state(self, letter_id=None):
        """Initialize or reinitialize the instance state."""
        # Import config here to avoid circular imports
        from config import MODEL_PARAMS, OPENROUTER_API_KEY, OPENROUTER_BASE_URL

        # Determine provider based on model
        provider = MODEL_PARAMS.get(self.model_name, {}).get('provider', 'openai')

        if provider == 'openrouter':
            # Create OpenRouter client
            self.client = openai.OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL
            )
        else:
            # Create standard OpenAI client
            self.client = openai.OpenAI(api_key=self.api_key)
            self.extra_headers = {}

        self.model = self.model_name
        self.temperature = self.base_temperature
        self.seed = self.base_seed
        self.messages = []
        self.parser = LLMOutputParser(similarity_threshold=0.9, model_name = "en_core_web_lg")
        # Setup new output management
        self.letter_id = letter_id or str(uuid.uuid4())[:8]
        self.output_dir = self.base_output_dir / self.letter_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Setup new logging
        self.setup_logging()
        # Initialize new run data
        self.run_data = {
            "raw_letter": None,
            "preprocessed_letter": None,
            "zero_shot_extraction": None,
            "initial_extraction": None,
            "initial_dataframe": None,
            "review": None,
            "final_extraction": None,
            "final_dataframe": None,
            "example_letter_path": str(self.example_letter_path),
            "run_metadata": {
                "model": self.model,
                "temperature": self.temperature,
                "seed": self.seed,
                "timestamp": datetime.now().isoformat(),
                "letter_id": self.letter_id,
                "session_id": str(uuid.uuid4())
            }
        }

    def reset(self, letter_id=None):
        """
        Reset the instance completely for a new letter analysis.

        Args:
            letter_id (str, optional): New letter ID for the reset instance
        """
        self.logger.info(
            "Resetting instance for new analysis. Old letter_id: %s, New letter_id: %s",
            self.letter_id,
            letter_id or 'auto-generated'
        )
        # Save current state if exists
        if hasattr(self, 'run_data') and self.run_data.get("raw_letter"):
            self.save_final_state()
        # Reinitialize state
        self.initialize_state(letter_id)
        session_id = self.run_data['run_metadata']['session_id']
        self.logger.info("Instance reset complete. New session ID: %s", session_id)
        return self

    def save_final_state(self):
        """Save the final state of the current analysis."""
        if hasattr(self, 'output_dir'):
            final_state_file = self.output_dir / "final_state.json"
            state_data = {
                "run_data": self.run_data,
                "final_messages": self.messages
            }
            with open(final_state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2)

    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "extraction_run.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"ChatGPT_{self.letter_id}")

    def save_conversation_log(self):
        """Save the complete conversation history."""
        conversation_file = self.output_dir / "conversation_log.json"
        with open(conversation_file, "w", encoding='utf-8') as f:
            json.dump(self.messages, f, indent=2)

    def save_raw_response(self, response_text):
        """Save the complete raw response to a file for analysis."""
        import time
        timestamp = int(time.time())
        raw_responses_dir = self.output_dir / "raw_responses"
        raw_responses_dir.mkdir(exist_ok=True)
        response_file = raw_responses_dir / f"raw_response_{timestamp}.txt"
        with open(response_file, "w", encoding='utf-8') as f:
            f.write(f"Response length: {len(response_text)} characters\n")
            f.write("="*50 + "\n")
            f.write(response_text)
            f.write("\n" + "="*50 + "\n")

    def save_json_output(self, data, stage):
        """Save JSON output for a specific stage."""
        if data is None:
            data = {"labels": {}}  # Empty JSON structure
            self.logger.warning("Empty JSON created for %s", str(stage))
        output_file = self.output_dir / f"{stage}.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def save_dataframe(self, df, fhand='example_results'):
        """Save DataFrame in multiple formats."""
        base_path = self.output_dir / fhand
        df.to_csv(f"{base_path}.csv", index=False)
        df.to_excel(f"{base_path}.xlsx", index=False)

    @staticmethod
    def preprocess_clinical_letter(letter_json: dict) -> str:
        """Preprocess clinical letter JSON into formatted string."""
        exclude_keys = {'url', 'medical specialty', 'sample name', 'keywords'}
        sections = []
        for key, value in letter_json.items():
            if key not in exclude_keys:
                display_title = key.replace('_', ' ').title()
                if value and str(value).strip():
                    sections.append(f"{display_title}:\n{value}\n")
        return "\n".join(sections)

    def load_letter(self, file_path: str | Path, save=True) -> str:
        """
        Load and preprocess a clinical letter from JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            str: Preprocessed letter text
        """
        file_path = Path(file_path)
        self.logger.info("Loading letter from: %s", str(file_path))
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                letter_json = json.load(f)
            if save is True:
                # Store raw letter
                self.run_data["raw_letter"] = letter_json
                # Preprocess letter
                preprocessed_letter = self.preprocess_clinical_letter(letter_json)
                self.run_data["preprocessed_letter"] = preprocessed_letter
                # Save both raw and preprocessed versions
                with open(self.output_dir / "raw_letter.json", 'w', encoding='utf-8') as f:
                    json.dump(letter_json, f, indent=2)
                with open(self.output_dir / "preprocessed_letter.txt", 'w', encoding='utf-8') as f:
                    f.write(preprocessed_letter)
                self.logger.info("Letter successfully loaded and preprocessed")
            else:
                preprocessed_letter = self.preprocess_clinical_letter(letter_json)
            return preprocessed_letter
        except Exception as e:
            self.logger.error("Error loading letter: %s", str(e))
            raise

    def set_pre_prompt(self, pre_prompt):
        """Sets the initial system message (pre-prompt) for the conversation."""
        self.messages = [{"role": "system", "content": pre_prompt}]

    def prompt(self, message, validate_json=False):
        """
        Enhanced prompt with logging and provider-specific handling.

        Args:
            message (str): The prompt message
            validate_json (bool): Whether to validate JSON response

        Returns:
            str or dict: Raw response or parsed JSON if validation requested
        """
        from config import MODEL_PARAMS

        self.logger.info("Sending prompt: %.100s...", message)
        self.messages.append({"role": "user", "content": message})
        try:
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": self.messages,
                "temperature": self.temperature,
                "max_tokens": MODEL_PARAMS.get(self.model, {}).get('max_tokens', 8192)
            }

            # Add seed for OpenAI models (not supported by all OpenRouter models)
            if MODEL_PARAMS.get(self.model, {}).get('provider') == 'openai':
                request_params["seed"] = self.seed

            # Add extra headers for OpenRouter
            if hasattr(self, 'extra_headers') and self.extra_headers:
                response = self.client.chat.completions.create(
                    extra_headers=self.extra_headers,
                    **request_params
                )
            else:
                response = self.client.chat.completions.create(**request_params)

            assistant_message = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": assistant_message})
            
            # Log the complete response
            self.logger.info("=== COMPLETE LLM RESPONSE START ===")
            self.logger.info("Response length: %d characters", len(assistant_message))
            self.logger.info("Full response:\n%s", assistant_message)
            self.logger.info("=== COMPLETE LLM RESPONSE END ===")

            # Save raw response to file for analysis
            self.save_raw_response(assistant_message)

            # Also log truncated version for readability
            self.logger.info("Received response: %.100s...", assistant_message)
            if validate_json:
                try:
                    parsed = self.parser.parse_llm_output(assistant_message)
                    self.logger.info("Successfully parsed JSON response")
                    return parsed
                except Exception as e:
                    self.logger.error("JSON validation failed: %s", str(e))
                    return self._handle_json_continuation(assistant_message)
            return assistant_message
        except Exception as e:
            self.logger.error("Prompt failed: %s", str(e))
            raise

    def _handle_json_continuation(self, failed_response, max_attempts=3):
        """
        Handle truncated JSON responses by implementing continuation logic.
        
        Args:
            failed_response (str): The response that failed validation
            max_attempts (int): Maximum number of continuation attempts

        Returns:
            dict: Complete parsed JSON response

        Raises:
            ValueError: If valid JSON cannot be obtained after all attempts
        """
        self.logger.info("Attempting JSON continuation logic for truncated response")
        
        # Try to parse partial results first
        try:
            partial_result = self.parser.parse_llm_output(failed_response, allow_partial=True)
            if partial_result.get("_partial"):
                last_key = partial_result.get("_last_key")
                self.logger.info("Found partial results with last complete entry: %s", last_key)
                
                # Collect all responses for merging
                all_responses = [{"labels": partial_result["labels"]}]
                
                # Continue from where we left off
                for attempt in range(max_attempts):
                    self.logger.info("Continuation attempt %d/%d", attempt + 1, max_attempts)
                    
                    continuation_prompt = self._create_continuation_prompt(
                        partial_result["labels"], last_key
                    )
                    
                    try:
                        # Get continuation response
                        continuation_response = self.prompt(continuation_prompt)
                        continuation_parsed = self.parser.parse_llm_output(continuation_response, allow_partial=True)
                        
                        if continuation_parsed.get("labels"):
                            all_responses.append(continuation_parsed)
                            self.logger.info("Successfully got continuation with %d new entries", 
                                           len(continuation_parsed["labels"]))
                            
                            # Check if this continuation is also partial
                            if continuation_parsed.get("_partial"):
                                last_key = continuation_parsed.get("_last_key")
                                self.logger.info("Continuation was also partial, last key: %s", last_key)
                                # Update for next iteration
                                partial_result = continuation_parsed
                            else:
                                # Complete response received
                                self.logger.info("Received complete continuation response")
                                break
                        else:
                            self.logger.warning("Continuation attempt returned no labels")
                            break
                            
                    except Exception as e:
                        self.logger.error("Continuation attempt %d failed: %s", attempt + 1, str(e))
                        if attempt == max_attempts - 1:
                            break
                
                # Merge all responses
                merged_result = self.parser.merge_json_responses(all_responses)
                self.logger.info("Merged %d responses into final result with %d total entries",
                               len(all_responses), len(merged_result["labels"]))
                return merged_result
                
        except Exception as e:
            self.logger.error("Partial parsing failed: %s", str(e))
        
        # Fallback to simple retry if continuation fails
        self.logger.info("Falling back to simple JSON retry")
        return self._simple_retry_json_extraction(failed_response)
    
    def _create_continuation_prompt(self, existing_labels: dict, last_key: str) -> str:
        """
        Create a prompt for the LLM to continue from where it left off.
        
        Args:
            existing_labels: Already extracted labels
            last_key: The last successfully extracted label key
            
        Returns:
            str: Continuation prompt
        """
        existing_count = len(existing_labels)
        
        prompt = f'''
        Your previous response was truncated. You have already successfully extracted {existing_count} annotations, with the last complete entry being "{last_key}".

        Please continue the analysis from where you left off and provide the remaining annotations in the same JSON format. 
        
        Start with any annotations that come after "{last_key}" and continue until you have extracted all remaining relevant clinical information.
        
        Return ONLY the JSON structure with the new annotations:
        
        {{
          "labels": {{
            "next_annotation_term": {{
              "context": "...",
              "qualifier": "...",
              "laterality": "...",
              "presence": "...",
              "primary_secondary": "...",
              "experiencer": "...", 
              "treatment_stage": "...",
              "snomed_ct": "..."
            }}
          }}
        }}
        '''
        
        return prompt
    
    def _simple_retry_json_extraction(self, failed_response, max_retries=1):
        """
        Simple fallback retry for JSON extraction.
        """
        retry_prompt = f'''
        The previous response was not in valid JSON format. 
        Please reformat the following content as a valid JSON object 
        following the specified structure exactly:

        {failed_response}
        '''

        for _ in range(max_retries):
            try:
                response = self.prompt(retry_prompt)
                return self.parser.parse_llm_output(response)
            except Exception as e:
                self.logger.error("Simple retry failed: %s", str(e))
        raise ValueError("Failed to obtain valid JSON after retries")

    def in_context_learning(self):
        """
        Perform in-context learning using example letter

        Args:
            preprocessed_example_letter (str, optional): Pre-processed example letter text.
                If None, automatically loads from self.example_letter_path
        """
        self.logger.info("Starting in-context learning with example letter")
        try:
            # load and pre-process example letter
            preprocessed_example_letter = self.load_letter(self.example_letter_path, save=False)
            example_letter_prompt = create_letter_prompt(preprocessed_example_letter)
            zero_shot_extraction = self.prompt(
                example_letter_prompt,
                validate_json=True
            )
            self.save_json_output(zero_shot_extraction, "zero_shot_extraction")
            # provide feedback
            feedback_prompt = create_feedback_prompt(self.example_annotation_path)
            _ = self.prompt(
                feedback_prompt,
                validate_json=False
            )
        except Exception as e:
            self.logger.error("In-context learning failed: %s", str(e))
            raise

    def analyze_clinical_letter(self, preprocessed_letter):
        """Enhanced analysis with comprehensive output management."""
        self.logger.info("Starting analysis for letter %s", str(self.letter_id))
        try:
            # Initial extraction
            self.logger.info("Performing initial extraction")
            letter_prompt = create_letter_prompt(preprocessed_letter)
            initial_extraction = self.prompt(
                letter_prompt,
                validate_json=True
            )
            self.run_data["initial_extraction"] = initial_extraction
            self.save_json_output(initial_extraction, "initial_extraction")
            # Create final DataFrame
            self.logger.info("Creating final DataFrame")
            print(initial_extraction)
            initial_df = self.parser.to_dataframe(initial_extraction)
            self.run_data["initial_dataframe"] = initial_df
            self.save_dataframe(initial_df, fhand='initial_output')
            # Save conversation log
            self.save_conversation_log()
            return self.run_data
        except Exception as e:
            self.logger.error("Analysis failed: %s", str(e))
            raise
        finally:
            # Save run metadata
            with open(self.output_dir / "run_metadata.json", "w", encoding='utf-8') as f:
                json.dump(self.run_data["run_metadata"], f, indent=2)

    def get_last_response(self):
        """Returns the last response from the assistant."""
        if self.messages and self.messages[-1]["role"] == "assistant":
            return self.messages[-1]["content"]
        return None

    def _try_alternative_search(self, row: pd.Series, idx: int, searcher: SnomedSearcher,
                            snomed_output_dir: Path) -> tuple[dict, bool]:
        """
        Attempt alternative search using LLM suggestion when initial search returns no results.
        Performs search without semantic tag filtering to maximize potential matches.

        Args:
            row: DataFrame row containing the annotation
            idx: Row index
            searcher: Initialized SnomedSearcher instance
            snomed_output_dir: Directory for saving search results

        Returns:
            tuple: (search_results, success_flag)
        """
        self.logger.info(f"Attempting alternative search for '{row['text']}'")

        # Create prompt for alternative search term
        prompt = SNOMED_SEARCH_REFINEMENT_PROMPT.format(
            text=row['text'],
            context=row['context']
        )

        # Get LLM suggestion
        alternative_term = self.prompt(prompt).strip()
        self.logger.info(f"LLM suggested alternative term: '{alternative_term}'")

        # Search with alternative term without semantic tag filtering
        search_results = searcher.search_term(alternative_term)

        # Create simplified results
        simplified_results = searcher._create_simplified_results(search_results)

        # Save results with _llmSearch suffix
        safe_text = row['text'].replace('/', '_').replace('\\', '_')
        llm_search_file = snomed_output_dir / f"{idx}_{safe_text}_llmSearch.json"

        with open(llm_search_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, indent=2, ensure_ascii=False)

        return simplified_results, bool(simplified_results.get('items', []))

    def process_snomed_codes(self):
        """Process SNOMED codes for each annotation in the initial dataframe."""
        self.logger.info("Starting SNOMED code processing")

        # Create SNOMED searcher
        searcher = SnomedSearcher()

        # Create output directory within the current output directory
        snomed_output_dir = self.output_dir / "snowstorm_results"

        try:
            # Process dataframe with SNOMED searcher
            searcher.process_dataframe(
                self.run_data["initial_dataframe"],
                output_dir=str(snomed_output_dir)
            )

            # Store conversation state after initial analysis
            analysis_state = self.messages.copy()

            # Initialize list to store codes for all rows
            updated_codes = []
            total_rows = len(self.run_data["initial_dataframe"])

            # Process each annotation
            for idx in range(total_rows):
                row = self.run_data["initial_dataframe"].iloc[idx]

                # Load SNOMED results for this annotation
                safe_text = row['text'].replace('/', '_').replace('\\', '_')
                results_file = snomed_output_dir / f"{idx}_{safe_text}.json"

                if results_file.exists():
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            snomed_results = json.load(f)

                        # Check if there are actual results
                        if snomed_results.get('items') and len(snomed_results['items']) > 0:
                            search_results = snomed_results
                            success = True
                        else:
                            # Try alternative search if initial search returns no results
                            search_results, success = self._try_alternative_search(
                                row, idx, searcher, snomed_output_dir
                            )

                        if success:
                            # Format SNOMED options for prompt
                            options_text = "\n".join(
                                f"- ID: {item['id']}, Term: {item['term']}"
                                for item in search_results['items']
                            )

                            # Reset conversation to post-analysis state
                            self.messages = analysis_state.copy()

                            # Create prompt with context
                            prompt = SNOMED_CODE_SELECTION_PROMPT.format(
                                text=row['text'],
                                context=row['context'],
                                qualifier=row['qualifier'],
                                snomed_options=options_text
                            )

                            # Get LLM selection
                            response = self.prompt(prompt)

                            # Clean response
                            code = response.strip()
                            if code.lower() == 'nan' or not code.isdigit():
                                code = None
                        else:
                            self.logger.info(
                                f"No SNOMED results found for annotation {idx} "
                                f"(both original and LLM-suggested searches failed)"
                            )
                            code = None
                    except (json.JSONDecodeError, IOError) as e:
                        self.logger.warning(
                            f"Error reading SNOMED results for annotation {idx}: {str(e)}"
                        )
                        code = None
                else:
                    self.logger.warning(
                        f"No SNOMED results file found for annotation {idx}"
                    )
                    code = None

                # Append code for this row
                updated_codes.append(code)

            # Verify we have the correct number of codes
            if len(updated_codes) != total_rows:
                raise ValueError(f"Number of processed codes ({len(updated_codes)}) "
                            f"does not match number of rows ({total_rows})")

            # Update dataframe with selected codes
            df = self.run_data["initial_dataframe"].copy()
            # Ensure we keep all rows by explicitly setting None values
            df.loc[:, 'snomed_ct'] = pd.Series(updated_codes, index=df.index)
            self.run_data["final_dataframe"] = df

            # Save updated dataframe
            self.save_dataframe(
                self.run_data["final_dataframe"],
                fhand='final_output'
            )

            self.logger.info("SNOMED code processing completed")

        except Exception as e:
            self.logger.error(f"Error processing SNOMED codes: {str(e)}")
            raise

    def full_letter_annotation(self, preprocessed_letter):
        """
        Perform complete letter annotation process including example prompting,
        in-context learning, final analysis and SNOMED code processing.
        """
        # Step 1: prompt LLM with example
        self.logger.info("Starting conversation for letter: %s", str(self.letter_id))
        try:
            _ = self.prompt(
                EXAMPLE_LETTER_PROMPT,
                validate_json=False
            )
        except Exception as e:
            self.logger.error("Example prompt failed: %s", str(e))
            raise

        # Step 2: run in-context learning
        self.in_context_learning()

        # Step 3: Run annotation on letter of interest
        letter_i_data = self.analyze_clinical_letter(preprocessed_letter)

        # Step 4: Process SNOMED codes
        self.process_snomed_codes()

        return self.run_data

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.save_final_state()
        self.reset()
