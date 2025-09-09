#!/usr/bin/env python3

"""
Script to run a comparison between human annotations and LLM annotations.

This script sets up the necessary directory structure to compare human annotations
against one or more LLM annotation sets.

Example:
    $ python run_human_comparison.py --llm_paths ../../results/annotations/llm/openAI/gpt-4o

Author: Liam Barrett
Version: 1.0.0
"""

import argparse
import logging
from pathlib import Path
import tempfile
import sys

from human_annotation_adapter import HumanAnnotationAdapter
from semantic_analysis import SemanticSimilarityAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default values
DEFAULT_HUMAN_PATH = "../../results/annotations/human"
DEFAULT_OUTPUT_ROOT = "../../results/similarity"

def run_human_comparison(
    human_path: str,
    llm_paths: list,
    output_root: str,
    annotator_id: str = None,
    force: bool = False
):
    """Run a comparison between human annotations and one or more LLM annotation sets.
    
    Args:
        human_path: Path to human annotations
        llm_paths: List of paths to LLM annotation directories
        output_root: Root directory for storing comparison results
        annotator_id: Specific human annotator ID to use
        force: If True, rerun comparisons even if output files exist
    """
    # Create output root directory if it doesn't exist
    Path(output_root).mkdir(exist_ok=True, parents=True)
    
    # Create the human annotation adapter
    adapter = HumanAnnotationAdapter(human_path)
    
    # Create the semantic similarity analyzer
    analyzer = SemanticSimilarityAnalyzer()
    
    # Create a temporary directory for the human annotations in LLM format
    with tempfile.TemporaryDirectory() as temp_dir:
        human_model_path = Path(temp_dir) / "human"
        
        # Create the mock LLM structure
        adapter.create_mock_llm_structure(str(human_model_path), annotator_id)
        
        # Run the comparison for each LLM
        for llm_path in llm_paths:
            llm_name = Path(llm_path).name
            comparison_id = f"human_vs_{llm_name}"
            
            logger.info(f"Running comparison: {comparison_id}")
            
            # Create comparison-specific output directory
            output_dir = Path(output_root) / comparison_id
            output_dir.mkdir(exist_ok=True)
            
            # Check if output already exists
            similarity_file = output_dir / f"similarity_human_vs_{llm_name}.csv"
            detailed_file = output_dir / f"detailed_matches_human_vs_{llm_name}.json"
            
            if not force and similarity_file.exists() and detailed_file.exists():
                logger.info(f"Skipping existing comparison: {comparison_id}")
                continue
            
            try:
                analyzer.compare_annotations(str(human_model_path), llm_path, str(output_dir))
                logger.info(f"Completed comparison: {comparison_id}")
            except Exception as e:
                logger.error(f"Error comparing {comparison_id}: {str(e)}")

def main():
    """Parse command-line arguments and run the comparison."""
    parser = argparse.ArgumentParser(
        description='Run a comparison between human annotations and LLM annotations'
    )
    parser.add_argument('--human_path', default=DEFAULT_HUMAN_PATH,
                      help='Path to human annotations')
    parser.add_argument('--llm_paths', nargs='+', required=True,
                      help='Paths to LLM annotation directories')
    parser.add_argument('--output_root', default=DEFAULT_OUTPUT_ROOT,
                      help='Root directory for storing comparison results')
    parser.add_argument('--annotator_id', 
                      help='Specific human annotator ID to use')
    parser.add_argument('--force', action='store_true',
                      help='Force rerun of existing comparisons')

    args = parser.parse_args()
    
    run_human_comparison(
        human_path=args.human_path,
        llm_paths=args.llm_paths,
        output_root=args.output_root,
        annotator_id=args.annotator_id,
        force=args.force
    )

if __name__ == "__main__":
    main()