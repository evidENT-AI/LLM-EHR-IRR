#!/usr/bin/env python3

"""Module for adapting human annotations to work with the semantic similarity analyzer.

Provides functionality to read and normalize human annotations to match the structure
expected by the semantic similarity analyzer for comparison with LLM annotations.

Author: Liam Barrett
Version: 1.0.0
"""

import logging
from pathlib import Path
import re
from typing import List, Optional
import pandas as pd

class HumanAnnotationAdapter:
    """Adapter for human annotations to make them compatible with SemanticSimilarityAnalyzer.
    
    This class reads human annotations from their source format and directory structure
    and transforms them to match the expected structure of LLM annotations.
    
    Args:
        human_annotations_root (str): Path to the root directory containing human annotations
    """
    
    def __init__(self, human_annotations_root: str):
        """Initialize the adapter with the path to human annotations.
        
        Args:
            human_annotations_root (str): Path to the root directory containing human annotations
        """
        self.human_annotations_root = Path(human_annotations_root)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized HumanAnnotationAdapter with root: {human_annotations_root}")
        
        # Column mapping from human to LLM format
        self.column_mapping = {
            'text': 'text',
            'context': 'context',
            'qualifier': 'qualifier',
            'laterality': 'laterality',
            'presence': 'presence',
            'presentation': 'primary_secondary',  # Map presentation to primary_secondary
            'experiencer': 'experiencer',
            'stage': 'treatment_stage',           # Map stage to treatment_stage
            'conceptIds': 'snomed_ct'             # Map conceptIds to snomed_ct
        }
        
        # Find all available annotators and letters
        self._discover_annotations()

    def _discover_annotations(self) -> None:
        """Discover all available annotators and their letter annotations."""
        self.annotators = []
        self.available_letters = set()
        
        for path in self.human_annotations_root.glob("annotator*"):
            if path.is_dir():
                self.annotators.append(path.name)
                
                # Find all letter annotations for this annotator
                for letter_file in path.glob("letter_*_annotations.csv"):
                    letter_id = re.search(r'letter_(\d+)_annotations', letter_file.name)
                    if letter_id:
                        self.available_letters.add(f"letter_{letter_id.group(1)}")
        
        self.logger.info(f"Found {len(self.annotators)} annotators with {len(self.available_letters)} unique letters")

    def get_available_letters(self) -> List[str]:
        """Get list of available letter IDs.
        
        Returns:
            List[str]: List of letter IDs (e.g., "letter_0001")
        """
        return sorted(list(self.available_letters))

    def get_annotations_for_letter(self, letter_id: str, annotator_id: Optional[str] = None) -> pd.DataFrame:
        """Get human annotations for a specific letter.
        
        Args:
            letter_id (str): Letter ID (e.g., "letter_0001")
            annotator_id (Optional[str]): Specific annotator ID to use. If None, uses the first available.
        
        Returns:
            DataFrame: Normalized annotations in LLM-compatible format
        """
        # Extract the numeric part of the letter ID
        match = re.search(r'letter_(\d+)', letter_id)
        if not match:
            raise ValueError(f"Invalid letter ID format: {letter_id}")
        
        letter_num = match.group(1)
        letter_filename = f"letter_{letter_num.zfill(4)}_annotations.csv"
        
        # Determine which annotator to use
        if annotator_id is None:
            # Find the first annotator who has this letter
            for annotator in self.annotators:
                annotation_path = self.human_annotations_root / annotator / letter_filename
                if annotation_path.exists():
                    annotator_id = annotator
                    break
            
            if annotator_id is None:
                raise FileNotFoundError(f"No annotator found with annotations for {letter_id}")
        else:
            if annotator_id not in self.annotators:
                raise ValueError(f"Annotator {annotator_id} not found")
        
        # Load the annotations
        annotation_path = self.human_annotations_root / annotator_id / letter_filename
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotations for {letter_id} not found for annotator {annotator_id}")
        
        self.logger.info(f"Loading annotations from {annotation_path}")
        df = pd.read_csv(annotation_path)
        
        # Normalize the DataFrame to match LLM annotation format
        return self._normalize_dataframe(df)
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize human annotation DataFrame to match LLM annotation format.
        
        Args:
            df (DataFrame): Original human annotations DataFrame
        
        Returns:
            DataFrame: Normalized DataFrame with columns matching LLM format
        """
        # Create a new DataFrame with the columns we need
        normalized_df = pd.DataFrame()
        
        # Map the columns
        for human_col, llm_col in self.column_mapping.items():
            if human_col in df.columns:
                normalized_df[llm_col] = df[human_col]
            else:
                normalized_df[llm_col] = None
        
        # Handle SNOMED CT codes - split conceptIds on semicolon if needed
        if 'conceptIds' in df.columns:
            normalized_df['snomed_ct'] = df['conceptIds'].apply(
                lambda x: x.split(';')[0] if pd.notna(x) and ';' in str(x) else x
            )
        
        # Fill NAs with empty strings
        normalized_df = normalized_df.fillna('')
        
        return normalized_df
    
    def create_mock_llm_structure(self, output_dir: str, annotator_id: Optional[str] = None) -> None:
        """Create a mock LLM directory structure for human annotations.
        
        This creates a directory structure matching the LLM format for compatibility
        with the existing semantic similarity analyzer.
        
        Args:
            output_dir (str): Directory to write the mock structure
            annotator_id (Optional[str]): Specific annotator to use. If None, uses first available for each letter.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        for letter_id in self.available_letters:
            # Create letter directory
            letter_dir = output_path / letter_id
            letter_dir.mkdir(exist_ok=True)
            
            try:
                # Get annotations and save as final_output.csv
                df = self.get_annotations_for_letter(letter_id, annotator_id)
                df.to_csv(letter_dir / "final_output.csv", index=False)
                self.logger.info(f"Created {letter_dir / 'final_output.csv'}")
            except (FileNotFoundError, ValueError) as e:
                self.logger.warning(f"Skipping {letter_id}: {str(e)}")
