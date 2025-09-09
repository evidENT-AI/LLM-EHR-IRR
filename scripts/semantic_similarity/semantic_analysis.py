#!/usr/bin/env python3

"""Module for analyzing semantic similarity between medical annotation datasets.

Uses BERT embeddings to compare medical annotations across different models,
calculating similarities between text fields and exact matches for categorical attributes.

Author: Liam Barrett
Version: 1.0.0
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple
import re

from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

class SemanticSimilarityAnalyzer:
    """Analyzer for comparing medical annotations using BERT embeddings.
    
    Uses Bio_ClinicalBERT by default to generate embeddings for medical text comparison
    and implements the Hungarian algorithm for optimal matching between annotation sets.

    Args:
        model_name (str): Name of the pretrained BERT model to use. 
            Defaults to "emilyalsentzer/Bio_ClinicalBERT"
    """
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading %s", str(model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def preprocess_text(self, text: str) -> str:
        """Standardize text for comparison"""
        text = str(text).lower().strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s-]', '', text)  # Remove punctuation except hyphens
        return text

    def get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate BERT embeddings for a list of texts.

        Args:
            texts (List[str]): List of text strings to embed

        Returns:
            np.ndarray: Matrix of BERT embeddings
        """
        # Preprocess texts
        texts = [self.preprocess_text(text) for text in texts]

        encoded = self.tokenizer(texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt")

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    def optimal_annotation_matching(
            self, df1: pd.DataFrame, df2: pd.DataFrame
        ) -> List[Tuple[int, int, float]]:
        """Find optimal matches between two sets of annotations using the Hungarian algorithm.

        Args:
            df1 (DataFrame): First set of annotations
            df2 (DataFrame): Second set of annotations

        Returns:
            List[Tuple[int, int, float]]: Matched annotation pairs with similarity scores
                above 0.5 as (index1, index2, similarity) tuples
        """
        # Define column weights for similarity calculation
        weights = {
            'text': 0.3,
            'context': 0.3,
            'qualifier': 0.1,
            'laterality': 0.05,
            'presence': 0.05,
            'primary_secondary': 0.05,
            'experiencer': 0.05,
            'treatment_stage': 0.05,
            'snomed_ct': 0.0
        }

        combined_sim = np.zeros((len(df1), len(df2)))

        # Calculate similarity for text and context using embeddings
        for col in ['text', 'context']:
            texts1, texts2 = df1[col].fillna('').tolist(), df2[col].fillna('').tolist()
            emb1 = self.get_bert_embeddings(texts1)
            emb2 = self.get_bert_embeddings(texts2)
            sim = cosine_similarity(emb1, emb2)
            combined_sim += weights[col] * sim

        # Calculate exact match similarities for categorical columns
        categorical_cols = ['qualifier', 'laterality', 'presence', 'primary_secondary',
                        'experiencer', 'treatment_stage', 'snomed_ct']

        for col in categorical_cols:
            if col in df1.columns and col in df2.columns:
                vals1 = df1[col].fillna('NA')
                vals2 = df2[col].fillna('NA')
                match_matrix = np.zeros((len(df1), len(df2)))

                for i, v1 in enumerate(vals1):
                    for j, v2 in enumerate(vals2):
                        match_matrix[i, j] = 1 if v1 == v2 else 0

                combined_sim += weights[col] * match_matrix

        # Find optimal matching using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(1 - combined_sim)

        # Return matches above threshold
        return [(i, j, combined_sim[i, j])
                for i, j in zip(row_ind, col_ind)
                if combined_sim[i, j] > 0.5]

    def load_annotations(self, model_path: Path) -> Dict[str, pd.DataFrame]:
        """Load annotation CSV files from the model output directory.

        Args:
            model_path (Path): Path to model output directory containing letter_* subdirectories

        Returns:
            Dict[str, DataFrame]: Dictionary mapping letter IDs to their annotation DataFrames,
                                sorted by letter number
        """
        # Get all valid letter directories
        letter_dirs = [
            d for d in model_path.glob("letter_*")
            if d.is_dir() and
            (d / "final_output.csv").exists() and
            d.name != "letter_0080"
        ]

        # Sort directories by letter number
        letter_dirs.sort(key=lambda x: int(x.name.split('_')[1]))

        # Create dictionary with sorted letters
        return {
            letter_dir.name: pd.read_csv(letter_dir / "final_output.csv")
            for letter_dir in letter_dirs
        }

    def compare_annotations(self,
                          model1_path: str,
                          model2_path: str,
                          output_dir: str = "similarity_results") -> pd.DataFrame:
        """Compare annotations between two model outputs and save results.

        Args:
            model1_path (str): Path to first model's output directory
            model2_path (str): Path to second model's output directory
            output_dir (str): Directory to save comparison results. Defaults to "similarity_results"

        Returns:
            DataFrame: Summary statistics for each letter's annotation comparison
        """
        self.logger.info("Comparing annotations between %s and %s", model1_path, model2_path)

        model1_path, model2_path = Path(model1_path), Path(model2_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        model1_annotations = self.load_annotations(model1_path)
        model2_annotations = self.load_annotations(model2_path)

        # Convert set to sorted list of letter IDs
        common_letters = sorted(set(model1_annotations.keys()) & set(model2_annotations.keys()),
                            key=lambda x: int(x.split('_')[1]))

        results = []
        detailed_matches = {}

        for letter_id in tqdm(common_letters):
            df1, df2 = model1_annotations[letter_id], model2_annotations[letter_id]
            matches = self.optimal_annotation_matching(df1, df2)

            avg_similarity = np.mean([sim for _, _, sim in matches]) if matches else 0
            n_matches = len(matches)

            results.append({
                'letter_id': letter_id,
                'similarity_score': avg_similarity,
                'n_matches': n_matches,
                'n_annotations_model1': len(df1),
                'n_annotations_model2': len(df2),
                'match_ratio': n_matches / max(len(df1), len(df2))
            })

            detailed_matches[letter_id] = {
                'matches': [{
                    'model1_text': df1.iloc[i]['text'],
                    'model2_text': df2.iloc[j]['text'],
                    'similarity': sim
                } for i, j, sim in matches]
            }

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / f"similarity_{model1_path.name}_vs_{model2_path.name}.csv",
                         index=False)

        with open(
            output_dir / f"detailed_matches_{model1_path.name}_vs_{model2_path.name}.json",
            'w', encoding='utf-8'
        ) as f:
            json.dump(detailed_matches, f, indent=2)

        return results_df
