#!/usr/bin/env python3

"""Script for comparing semantic similarity between medical annotation sets.

Initializes a similarity analyzer and processes annotation datasets from two model 
outputs, producing similarity scores and detailed match comparisons.

Example (generic):
   $ python run_semantic_analysis.py path/to/model1 path/to/model2 --output_dir results

Example (local):
   $ export GPT4O_PATH=../../results/pilot_cot/batch_run_20241126_215843/gpt-4o-2024-08-06
   $ export GPT4_MINI_PATH=../../results/pilot_cot/batch_run_20241126_121406/gpt-4o-mini-2024-07-18
   $ export RESULTS_PATH=../../results/pilot_cot/similarity_results
   $ python run_semantic_analysis.py $GPT4O_PATH $GPT4_MINI_PATH --output_dir $RESULTS_PATH

Author: Liam Barrett
Version: 1.0.0
"""

import argparse
from semantic_analysis import SemanticSimilarityAnalyzer

def main():
    """Run annotation comparison from command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare semantic similarity between model annotations'
    )
    parser.add_argument('model1_path', type=str, help='Path to first model annotations')
    parser.add_argument('model2_path', type=str, help='Path to second model annotations')
    parser.add_argument('--output_dir', type=str, default='similarity_results',
                        help='Directory to save results')
    args = parser.parse_args()

    analyzer = SemanticSimilarityAnalyzer()
    analyzer.compare_annotations(args.model1_path, args.model2_path, args.output_dir)

if __name__ == "__main__":
    main()
