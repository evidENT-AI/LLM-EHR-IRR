#!/usr/bin/env python3

"""
LLM Output Parser Module for Clinical Text Analysis

This module provides functionality for parsing, validating, and standardizing output 
from Large Language Models (LLMs) specifically focused on clinical text analysis.
It implements fuzzy matching, NLP-based similarity detection, and standardized
value mapping for clinical annotations.

The main class, LLMOutputParser, handles:
   - JSON extraction and validation from LLM outputs
   - Standardization of clinical annotations and attributes
   - Fuzzy matching using spaCy and thefuzz libraries
   - Value mapping for clinical terms and qualifiers
   - Conversion of parsed data to pandas DataFrames

Features:
   - NLP-powered similarity matching using spaCy models
   - Comprehensive value mapping for clinical terms
   - Special case handling for medical context
   - Fuzzy string matching for inexact matches
   - Detailed logging of parsing and matching processes
   - DataFrame conversion for downstream analysis

Example:
   ```python
   parser = LLMOutputParser(similarity_threshold=0.8, model_name="en_core_web_lg")
   
   # Parse LLM output
   parsed_data = parser.parse_llm_output(llm_response)
   
   # Convert to DataFrame
   df = parser.to_dataframe(parsed_data)
   ```

Dependencies:
   - spacy: For NLP and text similarity
   - pandas: For DataFrame operations
   - thefuzz: For fuzzy string matching
   - json: For JSON parsing
   - re: For regular expressions
   - typing: For type hints
   - logging: For operation logging

The module requires a spaCy model (preferably 'en_core_web_lg' for best results)
and will automatically download it if not present.

Author: Liam Barrett
Version: 1.0.0
"""

import logging
from typing import Dict, Optional, Any, Tuple
import re
import json
import pandas as pd
import spacy
from spacy.tokens import Doc
from thefuzz import fuzz

class LLMOutputParser:
    """
    Parser for extracting and validating JSON from LLM outputs with 
    enhanced value mapping and fuzzy matching.
    """
    def __init__(self, similarity_threshold: float = 0.8, model_name: str = "en_core_web_lg"):
        """
        Initialize parser with enhanced NLP capabilities.
        
        Args:
            similarity_threshold (float): Threshold for fuzzy matching (0-1)
            model_name (str): Name of spaCy model to use. Options:
                - en_core_web_lg (recommended): Large model with word vectors (~788MB)
                - en_core_web_md: Medium model with word vectors (~116MB)
                - en_core_web_sm: Small model without word vectors (~12MB)
        """
        self.logger = logging.getLogger(__name__)
        self.similarity_threshold = similarity_threshold
        # Load spaCy model
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            self.logger.warning("Downloading spaCy model '%s'...", model_name)
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        # Cache for Doc objects to avoid reprocessing
        self.doc_cache: Dict[str, Doc] = {}
        # Define expected fields and their types
        self.expected_fields = {
            "context": str,
            "qualifier": str,
            "laterality": str,
            "presence": str,
            "primary_secondary": str,
            "experiencer": str,
            "treatment_stage": str,
            "snomed_ct": str
        }
        # Standard values (original set)
        self.standard_values = {
            "qualifier": {
                "Sociodemographics",
                "Signs",
                "Symptoms", 
                "Diagnoses",
                "Treatments",
                "Procedures",
                "Risk Factors",
                "Test Results"
            },
            "laterality": {"left", "right", "bilateral", "NA"},
            "presence": {"confirmed", "suspected", "resolved", "negated", "NA"},
            "primary_secondary": {"primary", "secondary", "NA"},
            "experiencer": {"Patient", "Family", "NA"},
            "treatment_stage": {"pre-treatment", "current", "post-treatment", "NA"}
        }
        # Value mapping dictionaries
        self.value_mappings = {
            "qualifier": {
                # Sociodemographics
                "sociodemographic": "Sociodemographics", 
                "demographics": "Sociodemographics",
                "demographic": "Sociodemographics",
                # Signs
                "sign": "Signs",
                "physical finding": "Signs", 
                "clinical finding": "Signs",
                "observation": "Signs",
                # Symptoms
                "symptom": "Symptoms",
                "complaint": "Symptoms",
                "reported symptom": "Symptoms",
                # Diagnoses 
                "diagnosis": "Diagnoses",
                "condition": "Diagnoses",
                "medical condition": "Diagnoses", 
                "disease": "Diagnoses",
                # Treatments
                "treatment": "Treatments",
                "medication": "Treatments",
                "intervention": "Treatments",
                "therapy": "Treatments",
                # Procedures
                "procedure": "Procedures",
                "surgery": "Procedures",
                "operation": "Procedures",
                "surgical procedure": "Procedures",
                # Risk Factors
                "risk factor": "Risk Factors",
                "risk": "Risk Factors", 
                "health risk": "Risk Factors",
                # Test Results
                "test result": "Test Results",
                "lab result": "Test Results",
                "laboratory finding": "Test Results",
                "measurement": "Test Results",
                "vital sign": "Test Results"
            },
            "presence": {
                # Confirmed
                "present": "confirmed",
                "positive": "confirmed",
                "found": "confirmed",
                "observed": "confirmed",
                # Suspected
                "suspect": "suspected",
                "possible": "suspected",
                "probable": "suspected",
                # Resolved
                "past": "resolved",
                "historical": "resolved",
                "previous": "resolved",
                # Negated
                "absent": "negated",
                "negative": "negated",
                "none": "negated",
                "denied": "negated"
            },
            "treatment_stage": {
                # Pre-treatment
                "planned": "pre-treatment",
                "recommended": "pre-treatment",
                "scheduled": "pre-treatment",
                "pending": "pre-treatment",
                "proposed": "pre-treatment",
                # Current treatment
                "ongoing": "current",
                "active": "current",
                "in progress": "current",
                "current": "current",
                "undergoing": "current",
                # Post-treatment
                "completed": "post-treatment",
                "done": "post-treatment",
                "performed": "post-treatment", 
                "past": "post-treatment",
                "historical": "post-treatment"
            }
        }

    def find_closest_match(self, value: str, possible_values: set, field: str) -> Tuple[str, float]:
        """
        Find the closest matching standard value using NLP similarity.
        
        Args:
            value (str): Input value to match
            possible_values (set): Set of possible standard values
            field (str): Field name for context-specific processing
            
        Returns:
            Tuple[str, float]: (closest matching value, similarity score)
        """
        if not value or value.lower() == 'na':
            return "NA", 0.0
        best_match = "NA"
        highest_similarity = 0.0
        # Get spaCy Doc for input value
        if value not in self.doc_cache:
            self.doc_cache[value] = self.nlp(value.lower())
        value_doc = self.doc_cache[value]
        # First try exact token matching
        value_tokens = set(token.text.lower() for token in value_doc)
        for possible_value in possible_values:
            if possible_value.lower() in value_tokens:
                return possible_value, 1.0
        # Then try fuzzy string matching
        for possible_value in possible_values:
            # Get similarity score using multiple methods
            # spaCy similarity for semantic matching
            if possible_value not in self.doc_cache:
                self.doc_cache[possible_value] = self.nlp(possible_value.lower())
            spacy_similarity = value_doc.similarity(self.doc_cache[possible_value])
            # Levenshtein ratio for string similarity
            levenshtein_ratio = fuzz.ratio(value.lower(), possible_value.lower()) / 100
            # Token sort ratio for handling word order differences
            token_sort_ratio = fuzz.token_sort_ratio(value.lower(), possible_value.lower()) / 100
            # Combine similarity scores with weights
            combined_similarity = (
                0.4 * spacy_similarity +
                0.3 * levenshtein_ratio +
                0.3 * token_sort_ratio
            )
            # Apply field-specific boosts
            if field == "qualifier":
                # Boost matches for common medical terms
                medical_terms = {"symptom", "sign", "diagnosis", "treatment", "procedure"}
                if any(term in value.lower() for term in medical_terms):
                    combined_similarity *= 1.2
            if combined_similarity > highest_similarity:
                highest_similarity = combined_similarity
                best_match = possible_value
        return best_match, highest_similarity

    def standardize_value(self, field: str, value: str) -> str:
        """
        Map input values to standard values using mapping dictionaries and fuzzy matching.
        
        Args:
            field (str): The field name (e.g., "qualifier", "presence")
            value (str): The input value to standardize
            
        Returns:
            str: Standardized value or "NA" if no mapping found
        """
        if value is None or value.lower() == "na":
            return "NA"
        value = value.lower()
        # Check if the value is already in standard form
        if value.title() in self.standard_values.get(field, {}):
            return value.title()
        # Look up in mapping dictionary
        if field in self.value_mappings:
            standardized = self.value_mappings[field].get(value)
            if standardized:
                return standardized    
        # If no direct mapping found, try fuzzy matching
        if field in self.standard_values:
            closest_match, similarity = self.find_closest_match(
                value,
                self.standard_values[field],
                field
            )
            if similarity >= self.similarity_threshold:
                self.logger.info(
                    "Fuzzy matched '%s' to '%s' for field '%s' with similarity %.2f",
                    value,
                    closest_match,
                    field,
                    similarity
                )
                return closest_match
            else:
                self.logger.warning(
                    "No close match found for '%s' in field '%s'. "
                    "Best match was '%s' with similarity %.2f",
                    value,
                    field,
                    closest_match,
                    similarity
                )  
        # If still no match, try special case handling
        special_case_result = self._handle_special_cases(field, value)
        if special_case_result != "NA":
            return special_case_result
        return "NA"

    def _handle_special_cases(self, field: str, value: str) -> str:
        """Handle special cases for specific fields."""
        value = value.lower()
        if field == "experiencer":
            if any(term in value for term in ["patient", "self", "subject", "individual"]):
                return "Patient"
            elif any(term in value for term in ["family", "relative", "parent", "sibling", "mother", "father"]):
                return "Family"
        elif field == "primary_secondary":
            if any(term in value for term in ["primary", "main", "principal", "chief"]):
                return "primary"
            elif any(term in value for term in ["secondary", "additional", "associated", "other"]):
                return "secondary"   
        elif field == "laterality":
            if any(term in value for term in ["left", "l", "left-sided", "left side"]):
                return "left"
            elif any(term in value for term in ["right", "r", "right-sided", "right side"]):
                return "right"
            elif any(term in value for term in ["bilateral", "both", "two-sided", "both sides"]):
                return "bilateral"    
        return "NA"

    def validate_label_entry(self, label: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean a single label entry with enhanced value mapping and fuzzy matching."""
        cleaned_entry = {}
        # Check for required fields
        for field, _ in self.expected_fields.items():
            value = entry.get(field)
            # Handle missing fields
            if value is None:
                self.logger.debug(
                    "Missing field '%s' for label '%s'. Setting to 'NA'",
                    field,
                    label
                )
                cleaned_entry[field] = "NA"
                continue
            # Validate and standardize value
            if field in self.value_mappings or field in self.standard_values:
                standardized_value = self.standardize_value(field, str(value))
                if standardized_value == "NA" and value != "NA":
                    self.logger.warning(
                        "Could not map value '%s' for field '%s' in label '%s' "
                        "even with fuzzy matching. Setting to 'NA'",
                        value,
                        field,
                        label
                    )
                cleaned_entry[field] = standardized_value
            else:
                cleaned_entry[field] = str(value)
        return cleaned_entry

    def extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract and repair JSON string from LLM output text, handling markdown code blocks."""
        try:
            # First, try to extract from markdown code blocks
            markdown_match = re.search(r'```(?:json)?\s*(\{.*)\s*```', text, re.DOTALL)
            if markdown_match:
                json_str = markdown_match.group(1).rstrip()
                # Handle case where closing ``` might be missing due to truncation
                if not json_str.endswith('}'):
                    # Count braces to see if we need to close the JSON
                    open_braces = json_str.count('{') - json_str.count('}')
                    if open_braces > 0:
                        # Add missing closing braces
                        json_str += '}' * open_braces
                        self.logger.warning("Added %d missing closing braces to truncated JSON", open_braces)
                try:
                    # Try parsing the markdown-extracted JSON
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError as e:
                    # If it fails, continue with repair logic below
                    pass
            
            # Find text starting from first { (might not end with } if truncated)
            json_match = re.search(r'\{.*', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    # Try parsing as is
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError as e:
                    # Try fixing truncation issues
                    self.logger.info("JSON parse error: %s. Attempting repair...", str(e))
                    
                    # Get the position of the error
                    pos = getattr(e, 'pos', len(json_str))
                    
                    # Count unclosed braces to determine needed closing braces
                    open_braces = json_str[:pos].count('{') - json_str[:pos].count('}')
                    open_brackets = json_str[:pos].count('[') - json_str[:pos].count(']')
                    
                    # Try to fix the truncated JSON
                    fixed_json = json_str[:pos].rstrip()
                    
                    # Handle different truncation cases
                    if fixed_json.endswith(':'):
                        # Truncated after a key, add a placeholder value
                        fixed_json += ' "NA"'
                    elif fixed_json.endswith(','):
                        # Truncated after a comma, remove the trailing comma
                        fixed_json = fixed_json[:-1]
                    elif not fixed_json.endswith(('"', '}', ']')):
                        # If truncated mid-value, try to close the value
                        if '"' in fixed_json and fixed_json.count('"') % 2 == 1:
                            # Unclosed string
                            fixed_json += '"'
                    
                    # Close any open structures
                    fixed_json += '}' * open_braces + ']' * open_brackets
                    
                    self.logger.warning("Attempting to fix truncated JSON. Original error: %s", str(e))
                    
                    # Validate the fixed JSON
                    try:
                        json.loads(fixed_json)
                        return fixed_json
                    except Exception as fix_error:
                        self.logger.error("Failed to fix truncated JSON: %s", str(fix_error))
            return None
        except Exception as e:
            self.logger.error("Error extracting JSON: %s", str(e))
            return None

    def find_last_complete_entry(self, partial_json_str: str) -> tuple[str, str, dict]:
        """
        Programmatically find the last complete annotation entry in a truncated JSON string.
        
        Args:
            partial_json_str: The truncated JSON string
            
        Returns:
            tuple: (last_complete_key, continuation_point, partial_labels_dict)
        """
        try:
            # Remove markdown wrapper if present
            clean_json = partial_json_str.strip()
            if clean_json.startswith('```json'):
                clean_json = clean_json[7:].strip()
            if clean_json.endswith('```'):
                clean_json = clean_json[:-3].strip()
            
            # Try to parse the JSON incrementally by building a valid structure
            complete_entries = {}
            last_complete_key = None
            
            # Find the start of labels
            labels_pattern = r'"labels"\s*:\s*\{'
            labels_match = re.search(labels_pattern, clean_json)
            if not labels_match:
                return None, None, {}
            
            # Start from after "labels": {
            content_start = labels_match.end()
            content = clean_json[content_start:]
            
            # Build JSON incrementally by trying to parse smaller chunks
            brace_depth = 1  # We're already inside the labels object
            current_pos = 0
            entry_candidates = []
            
            # Find all potential entry boundaries
            in_string = False
            escape_next = False
            
            for i, char in enumerate(content):
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    
                if not in_string:
                    if char == '{':
                        brace_depth += 1
                    elif char == '}':
                        brace_depth -= 1
                        
                        # When we return to depth 1, we've completed an entry
                        if brace_depth == 1:
                            # Try to parse from the beginning up to this point
                            test_content = content[:i+1]
                            
                            # Ensure proper JSON structure
                            if not test_content.strip().endswith(','):
                                test_content = test_content.rstrip() + ','
                            
                            test_json = '{"labels": {' + test_content + '}}'
                            
                            try:
                                # Remove trailing comma for valid JSON
                                clean_test = test_json.replace(',}', '}')
                                parsed = json.loads(clean_test)
                                
                                if 'labels' in parsed and parsed['labels']:
                                    # Successfully parsed - update our complete entries
                                    complete_entries = parsed['labels']
                                    # Find the last key (they're ordered)
                                    if complete_entries:
                                        last_complete_key = list(complete_entries.keys())[-1]
                                        
                            except json.JSONDecodeError:
                                # This chunk isn't complete yet, continue
                                continue
            
            self.logger.info("Found %d complete entries, last key: %s", 
                           len(complete_entries), last_complete_key)
            
            return last_complete_key, content, complete_entries
            
        except Exception as e:
            self.logger.error("Error finding last complete entry: %s", str(e))
            return None, None, {}

    def merge_json_responses(self, responses: list[dict]) -> dict:
        """
        Merge multiple JSON responses into a single complete response.
        
        Args:
            responses: List of parsed JSON responses
            
        Returns:
            dict: Merged JSON response
        """
        merged_labels = {}
        
        for response in responses:
            if isinstance(response, dict) and 'labels' in response:
                merged_labels.update(response['labels'])
        
        return {"labels": merged_labels}

    def parse_llm_output(self, llm_output: str, allow_partial: bool = False) -> Dict[str, Dict[str, Any]]:
        """Parse and validate LLM output into structured format with duplicate handling."""
        try:
            # Extract and potentially repair JSON from text
            json_str = self.extract_json_from_text(llm_output)
            if not json_str:
                raise ValueError("No JSON found in output")
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Extract labels dictionary
            labels = parsed.get("labels", {})
            if not labels:
                raise ValueError("No labels found in JSON")
            
            # Handle duplicate entries - keep only the first occurrence of each key
            seen_keys = set()
            unique_labels = {}
            
            for label, entry in labels.items():
                # Skip duplicate entries
                if label in seen_keys:
                    self.logger.warning(f"Skipping duplicate entry for label: {label}")
                    continue
                
                seen_keys.add(label)
                
                try:
                    unique_labels[label] = self.validate_label_entry(label, entry)
                except Exception as e:
                    self.logger.error(f"Error processing label {label}: {str(e)}")
                    continue
            
            return {"labels": unique_labels}
            
        except json.JSONDecodeError as e:
            self.logger.error("JSON parsing error: %s", str(e))
            
            if allow_partial:
                # Try to extract partial results
                self.logger.info("Attempting to parse partial JSON...")
                last_key, content, partial_labels = self.find_last_complete_entry(llm_output)
                if partial_labels:
                    self.logger.info("Found %d complete entries, last entry: %s", 
                                   len(partial_labels), last_key)
                    return {"labels": partial_labels, "_partial": True, "_last_key": last_key}
            
            raise
        except Exception as e:
            self.logger.error("Error parsing LLM output: %s", str(e))
            raise
    def to_dataframe(self, parsed_json: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert parsed JSON from LLM output to a pandas DataFrame
        
        Args:
            parsed_json (dict): Dictionary containing extracted labels and attributes
            
        Returns:
            pd.DataFrame: DataFrame with labels and metadata
        """
        # Extract the labels dictionary if it exists
        labels_dict = parsed_json.get("labels", parsed_json)
        records = [
            {
                'text': text,
                **attributes
            }
            for text, attributes in labels_dict.items()
        ]
        df = pd.DataFrame(records)
        df = df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        df = df.replace(['na', 'NA', 'None', ''], None)
        return df
