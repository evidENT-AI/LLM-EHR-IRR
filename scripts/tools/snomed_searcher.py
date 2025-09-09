#!/usr/bin/env python
"""
Author: Liam Barrett
Version: 1.0.0

SNOMED CT Searcher Module

This module provides functionality to search SNOMED CT codes using a Snowstorm server
for clinical annotations. It is designed to process clinical text annotations and find 
corresponding SNOMED CT concepts, supporting semantic tag filtering based on the 
annotation type (e.g., symptoms, diagnoses, procedures).

Prerequisites:
    This module requires Elasticsearch and Snowstorm to be installed and running locally.
    If properly installed and loaded with UK SNOMED-CT, they can be activated using:

    1. Start Elasticsearch:
       ```
       cd /opt/elasticsearch
       ./bin/elasticsearch
       ```

    2. Start Snowstorm:
       ```
       java -Xms2g -Xmx4g -jar snowstorm-10.5.1.jar --snowstorm.rest-api.readonly=true
       ```

    Once both services are running, this module can connect to Snowstorm's API.

Output:
    The module creates both detailed and simplified JSON outputs for each search:
    - Detailed output includes full SNOMED CT concept information
    - Simplified output includes just concept IDs and terms

Example:
    >>> searcher = SnomedSearcher()
    >>> df = pd.DataFrame({
    ...     'text': ['otalgia'],
    ...     'qualifier': ['symptom'],
    ...     'context': ['patient reports otalgia']
    ... })
    >>> searcher.process_dataframe(df)

Dependencies:
    - pandas
    - requests
    - json
    - os
"""

import json
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
import requests
import pandas as pd

@dataclass
class SearchConfig:
    """Configuration for SNOMED CT search parameters."""
    server_url: str = "http://localhost:8080"
    timeout_seconds: int = 30
    semantic_tag_mapping: Dict[str, str] = None

    def __post_init__(self):
        """Initialize default semantic tag mapping if none provided."""
        if self.semantic_tag_mapping is None:
            self.semantic_tag_mapping = {
                'symptom': 'finding',
                'sign': 'finding',
                'diagnosis': 'disorder',
                'procedure': 'procedure',
                'test result': 'observable entity',
                'treatment': 'product'
            }
class SnomedSearcher:
    """
    A class to search SNOMED CT codes using Snowstorm server for clinical annotations.
    
    This class handles the interaction with a Snowstorm server to search for SNOMED CT
    concepts based on clinical annotations. It supports semantic tag filtering and
    produces both detailed and simplified search results.
    """

    def __init__(self, config: Optional[SearchConfig] = None):
        """
        Initialize the SnomedSearcher.
        
        Args:
            config: Search configuration parameters. If None, uses defaults.
        """
        self.config = config or SearchConfig()

    def _get_semantic_tag(self, qualifier: str) -> Optional[str]:
        """
        Map the qualifier to appropriate SNOMED CT semantic tag.
        
        Args:
            qualifier: The qualifier from the annotation
            
        Returns:
            The corresponding semantic tag or None if no mapping exists
        """
        # Convert qualifier to lowercase for consistent matching
        qualifier = qualifier.lower()

        # Return corresponding semantic tag if exists
        for key, value in self.config.semantic_tag_mapping.items():
            if key in qualifier:
                return value
        return None

    def search_term(self, term: str, semantic_tag: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for a term in Snowstorm.
        
        Args:
            term: The term to search for
            semantic_tag: Optional semantic tag to filter results
            
        Returns:
            The JSON response from Snowstorm
            
        Raises:
            requests.RequestException: If the request fails
        """
        # Construct base URL
        url = f"{self.config.server_url}/browser/MAIN/descriptions"

        # Build parameters
        params = {
            "term": term,
            "active": "true",
            "conceptActive": "true",
            "searchMode": "STANDARD"
        }

        # Add semantic tag if provided
        if semantic_tag:
            params["semanticTag"] = semantic_tag

        # Make request with timeout
        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.config.timeout_seconds
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error searching for term '{term}': {str(e)}")
            return {"items": []}  # Return empty results on error

    def _create_simplified_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create simplified version of search results containing only id and term.
        
        Args:
            results: Full search results from Snowstorm
            
        Returns:
            Simplified results dictionary
        """
        simplified = {"items": []}
        if "items" in results:
            for item in results["items"]:
                simplified["items"].append({
                    "term": item["term"],
                    "id": item["concept"]["conceptId"]
                })
        return simplified

    def process_dataframe(self, annotation_df: pd.DataFrame, output_dir: str = "snomed_results"):
        """
        Process each row in the dataframe and save SNOMED search results.
        Creates both full and simplified JSON outputs.
        
        Args:
            annotation_df: DataFrame containing the clinical annotations
            output_dir: Directory to save the results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Process each row
        for idx, row in annotation_df.iterrows():
            # Get semantic tag based on qualifier if present
            semantic_tag = None
            if pd.notna(row['qualifier']):
                semantic_tag = self._get_semantic_tag(row['qualifier'])

            # Search for the term
            results = self.search_term(row['text'], semantic_tag)

            # Create filenames
            safe_text = row['text'].replace('/', '_').replace('\\', '_')
            full_filename = f"{idx}_{safe_text}_full.json"
            simple_filename = f"{idx}_{safe_text}.json"

            # Create simplified version
            simplified_results = self._create_simplified_results(results)

            # Save full results
            with open(os.path.join(output_dir, full_filename), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Save simplified results
            with open(os.path.join(output_dir, simple_filename), 'w', encoding='utf-8') as f:
                json.dump(simplified_results, f, indent=2, ensure_ascii=False)


def example_usage():
    """Example usage of the SnomedSearcher class."""
    # Sample DataFrame
    data = {
        'text': ['3-year-old', 'otalgia', 'adenotonsillitis'],
        'context': [
            'this is a 3-year-old child with a history of adenotonsillitis.',
            'patient complains of otalgia',
            'diagnosed with adenotonsillitis'
        ],
        'qualifier': ['sociodemographics', 'symptom', 'diagnosis'],
        'laterality': ['', '', ''],
        'presence': ['', '', ''],
        'primary_secondary': ['', '', ''],
        'experiencer': ['patient', 'patient', 'patient'],
        'treatment_stage': ['', '', '']
    }
    example_df = pd.DataFrame(data)

    # Create searcher and process dataframe
    searcher = SnomedSearcher()
    searcher.process_dataframe(example_df, output_dir='./results/test_snomed_results')


if __name__ == "__main__":
    example_usage()
