#!/usr/bin/env python3

"""
Batch script for running semantic similarity analysis between multiple model annotation sets
and human annotations.

This script performs an exhaustive comparison of all model pairs including human annotations,
creating a structured output directory for each comparison.

Example:
    $ python batch_similarity_analysis.py --include_human

Author: Liam Barrett
Version: 2.0.0
"""

import itertools
import logging
from pathlib import Path
import argparse
import time
from typing import List, Dict, Optional
import tempfile

from semantic_analysis import SemanticSimilarityAnalyzer
from human_annotation_adapter import HumanAnnotationAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default model paths - these can be extended as needed
DEFAULT_MODEL_PATHS = [
    "../../results/annotations/llm/anthropic/claude-sonnet-3-5",
    "../../results/annotations/llm/google/gemini-pro-1-5",
    "../../results/annotations/llm/meta/llama3-1-405b",
    "../../results/annotations/llm/meta/llama3-1-70b",
    "../../results/annotations/llm/meta/llama-3-1-8b",
    "../../results/annotations/llm/openAI/gpt-4o",
]

# Default human annotation path
DEFAULT_HUMAN_PATH = "../../results/annotations/human"

# Default output directory
DEFAULT_OUTPUT_ROOT = "../../results/similarity"

def extract_model_name(path: str) -> str:
    """Extract the model name from its path.
    
    Args:
        path: Path to the model annotations
        
    Returns:
        The model name extracted from the path
    """
    return Path(path).name

def extract_annotator_id(path: str) -> str:
    """Extract the annotator ID from the human annotation path.
    
    Args:
        path: Path to the human annotations
        
    Returns:
        The annotator ID extracted from the path
    """
    if "human_annotator" in path:
        return path.split("human_annotator_")[-1]
    else:
        return "human"  # Default name if not using the annotator-specific format

def create_comparison_id(model1_path: str, model2_path: str) -> str:
    """Create a standardized comparison ID for two models.
    
    Args:
        model1_path: Path to the first model annotations
        model2_path: Path to the second model annotations
        
    Returns:
        A comparison ID string in the format "model1_vs_model2"
    """
    model1_name = extract_model_name(model1_path)
    model2_name = extract_model_name(model2_path)
    
    # Special handling for human paths
    if "human_annotator" in model1_path:
        model1_name = f"human-{extract_annotator_id(model1_path)}"
    if "human_annotator" in model2_path:
        model2_name = f"human-{extract_annotator_id(model2_path)}"
        
    return f"{model1_name}_vs_{model2_name}"

def setup_human_annotators(human_path: str) -> Dict[str, str]:
    """Set up human annotators and create temporary LLM-style directories.
    
    Args:
        human_path: Path to human annotations
        
    Returns:
        Dictionary mapping annotator IDs to their temporary paths
    """
    # Create adapter
    adapter = HumanAnnotationAdapter(human_path)
    
    # Create temporary directories for each annotator
    annotator_paths = {}
    
    # Option 1: Create a single "human" model with combined annotations
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "human"
        adapter.create_mock_llm_structure(str(temp_path))
        annotator_paths["human"] = str(temp_path)
        
        # Store the temporary directory so it doesn't get deleted
        annotator_paths["_temp_dir"] = temp_dir

    return annotator_paths

def run_all_comparisons(
    model_paths: List[str],
    output_root: str,
    include_human: bool = False,
    human_path: str = DEFAULT_HUMAN_PATH,
    skip_existing: bool = True,
    specific_pairs: Optional[List[Dict[str, str]]] = None
) -> None:
    """Run semantic similarity analysis for all model pairs.
    
    Args:
        model_paths: List of paths to model annotation directories
        output_root: Root directory for storing comparison results
        include_human: Whether to include human annotations in comparisons
        human_path: Path to human annotations
        skip_existing: If True, skip comparisons that already have result files
        specific_pairs: Optional list of specific model pairs to compare
                       Each pair is a dict with 'model1' and 'model2' keys
    """
    start_time = time.time()
    analyzer = SemanticSimilarityAnalyzer()

    # Create output root directory if it doesn't exist
    Path(output_root).mkdir(exist_ok=True, parents=True)
    
    all_paths = model_paths.copy()
    human_annotator_paths = {}
    
    # Add human annotators if requested
    if include_human:
        logger.info(f"Including human annotations from {human_path}")
        human_annotator_paths = setup_human_annotators(human_path)
        all_paths.append(human_annotator_paths["human"])

    # Generate all pairs of models to compare
    if specific_pairs:
        # Use specific pairs if provided
        pairs = [(pair['model1'], pair['model2']) for pair in specific_pairs]
    else:
        # Otherwise generate all possible pairs
        pairs = list(itertools.combinations(all_paths, 2))

    total_comparisons = len(pairs)
    logger.info(f"Starting batch analysis with {total_comparisons} comparisons")

    completed = 0
    skipped = 0

    for model1_path, model2_path in pairs:
        comparison_id = create_comparison_id(model1_path, model2_path)
        output_dir = Path(output_root) / comparison_id

        # Check if output already exists
        if skip_existing and output_dir.exists():
            model1_name = extract_model_name(model1_path)
            model2_name = extract_model_name(model2_path)
            
            # Handle human paths
            if "human_annotator" in model1_path:
                model1_name = f"human-{extract_annotator_id(model1_path)}"
            if "human_annotator" in model2_path:
                model2_name = f"human-{extract_annotator_id(model2_path)}"
                
            similarity_file = output_dir / f"similarity_{model1_name}_vs_{model2_name}.csv"
            detailed_file = output_dir / f"detailed_matches_{model1_name}_vs_{model2_name}.json"

            if similarity_file.exists() and detailed_file.exists():
                logger.info(f"Skipping existing comparison: {comparison_id}")
                skipped += 1
                continue

        # Create comparison-specific output directory
        output_dir.mkdir(exist_ok=True)

        logger.info(f"Running comparison: {comparison_id} ({completed+1}/{total_comparisons-skipped})")

        try:
            analyzer.compare_annotations(model1_path, model2_path, str(output_dir))
            completed += 1
        except Exception as e:
            logger.error(f"Error comparing {comparison_id}: {str(e)}")

    elapsed_time = time.time() - start_time
    logger.info(f"Batch analysis completed in {elapsed_time:.2f} seconds")
    logger.info(f"Completed: {completed}, Skipped: {skipped}, Total: {total_comparisons}")
    
    # Clean up temporary directories
    if "_temp_dir" in human_annotator_paths:
        # This will clean up the temporary directory
        pass

def main():
    """Parse command-line arguments and run batch analysis."""
    parser = argparse.ArgumentParser(
        description='Run batch semantic similarity analysis between multiple model annotation sets'
    )
    parser.add_argument('--model_paths', nargs='+', default=DEFAULT_MODEL_PATHS,
                      help='Paths to model annotation directories')
    parser.add_argument('--output_root', default=DEFAULT_OUTPUT_ROOT,
                      help='Root directory for storing comparison results')
    parser.add_argument('--force', action='store_true',
                      help='Force rerun of existing comparisons')
    parser.add_argument('--model_pair', nargs=2, action='append',
                      help='Specific model pair to compare (can be used multiple times)')
    parser.add_argument('--include_human', action='store_true',
                      help='Include human annotations in comparisons')
    parser.add_argument('--human_path', default=DEFAULT_HUMAN_PATH,
                      help='Path to human annotations')

    args = parser.parse_args()

    # Convert model_pair arguments to the format needed by run_all_comparisons
    specific_pairs = None
    if args.model_pair:
        specific_pairs = [{'model1': pair[0], 'model2': pair[1]} for pair in args.model_pair]

    run_all_comparisons(
        model_paths=args.model_paths,
        output_root=args.output_root,
        include_human=args.include_human,
        human_path=args.human_path,
        skip_existing=not args.force,
        specific_pairs=specific_pairs
    )

if __name__ == "__main__":
    main()
