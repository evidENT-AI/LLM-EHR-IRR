# NLP-EHR-IRR: LLM Evaluation Framework for Clinical Documentation

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the complete codebase for the research article: **"Developing an open-source framework for LLM evaluation of patients using EHR clinical documentation; performance of LLMs relative to medical professionals"**.

The framework evaluates how Large Language Models (LLMs) perform in extracting structured clinical information from Electronic Health Records (EHRs) compared to medical professionals. Using openly available Ear, Nose and Throat (ENT) clinical documentation from MTSamples, we assess both reliability and accuracy metrics through standardized SNOMED-CT coding.

## Key Features

- **Multi-model Support**: Evaluation framework for GPT-4o, Claude 3.5, Gemini 1.5 Pro, Gemma 3, and LLAMA variants
- **Standardized Comparison**: SNOMED-CT code assignment enabling standardized inter-rater reliability assessment
- **Comprehensive Analysis**: Bayesian hierarchical modelling for rigorous statistical comparison
- **Modular Architecture**: Easily extensible to new models and clinical domains
- **Complete Pipeline**: From raw clinical text to statistical analysis and visualization

## Project Structure

```
nlp_ehr/
├── data/                       # Clinical letters and annotations
│   ├── mt_samples/            # MTSamples ENT clinical letters
│   └── annotated_letter_0080.json  # Gold standard example
├── scripts/                    # Core implementation
│   ├── anthropic/             # Claude model integration
│   ├── google/                # Gemini/Gemma integration
│   ├── llama/                 # LLAMA model integration
│   ├── openai/                # GPT model integration
│   ├── semantic_similarity/   # Similarity analysis tools
│   └── tools/                 # Utility scripts
├── notebooks/                  # Analysis notebooks
│   ├── bayesian-hierarchical-in-analysis.ipynb
│   ├── chain_of_thought.ipynb
│   ├── get_mtsamples_data.ipynb
│   └── eval_metrics_gpt4o.ipynb
├── results/                    # Model outputs and analyses
│   ├── annotations/           # Model and human annotations
│   └── similarity/            # Comparison results
├── docs/                       # Documentation
└── LICENSE.txt                # GNU GPL v3
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Elasticsearch and Snowstorm (for SNOMED-CT searching)
- API keys for LLM services (OpenAI, Anthropic, Google Cloud)

### Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/nlp-ehr-irr.git
cd nlp-ehr-irr
```

2. **Create and activate a conda environment**

```bash
conda create -n nlp-ehr python=3.9
conda activate nlp-ehr
```

3. **Install dependencies**

```bash
# For development (editable install)
pip install -e .

# Or for regular installation
pip install .
```

4. **Set up SNOMED-CT services** (optional, for SNOMED code assignment)

```bash
# Start Elasticsearch
cd /opt/elasticsearch
./bin/elasticsearch

# Start Snowstorm (requires snowstorm jar file)
java -Xms2g -Xmx4g -jar snowstorm-10.5.1.jar --snowstorm.rest-api.readonly=true
```

5. **Configure API keys (optional)**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
PROJECT_ID=your_gcp_project_id
```

## Quick Start

### 1. Process a Single Clinical Letter

```python
from scripts.openai.labeller import ChatGPTInstance

# Initialize model
gpt = ChatGPTInstance(
    api_key="your-api-key",
    model="gpt-4o-2024-08-06",
    output_dir="results/annotations"
)

# Load and process letter
letter = gpt.load_letter("data/mt_samples/letters/letter_0001.json")
results = gpt.full_letter_annotation(letter)
```

### 2. Run Batch Processing

```bash
# Process multiple letters with a specific model
python scripts/openai/batch_run_labelling.py
```

## Core Components

### Model Integration Modules

Each model family has its own integration module:

- `scripts/openai/`: GPT models via OpenAI API
- `scripts/anthropic/`: Claude models via Anthropic API
- `scripts/google/`: Gemini/Gemma models via Google Vertex AI
- `scripts/llama/`: LLAMA models via Vertex AI

### Analysis Pipeline

1. **Letter Processing**: Load and preprocess clinical letters from JSON
2. **In-context Learning**: Provide example annotations for consistency
3. **Information Extraction**: Extract clinical information in 7 categories
4. **SNOMED Mapping**: Assign standardized medical codes
5. **Comparison Analysis**: Calculate inter-rater reliability metrics
6. **Statistical Analysis**: Bayesian hierarchical modeling for significance testing

## Citation

If you use this code in your research, please cite:

```bibtex
@article{barrett2025llm,
  title={Developing an open-source framework for LLM evaluation of patients using EHR clinical documentation; performance of LLMs relative to medical professionals},
  author={Barrett, Liam and Joshi, Nikhil and North, Alexander S. and Dimitrov, Lilia and Maughan, Elizabeth F. and Ross, Talisa and Pankhania, Rahul and Paramjothy, Kiara and Minty, Iona and Trajano, Luiza Farache and Smith, Sabrina L. and Mason, Katrina A. and Bhargava, Eishaan K. and Donnelly, Catherine and Fatoum, Hanaa and Padiyar, Ashwika and Kader, Zahra and Chan, Kevin and Schilder, Anne G.M. and Mehta, Nishchay},
  journal={NPJ Digital Medicine},
  year={2025},
  note={In preparation}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.txt](LICENSE.txt) file for details.

## Contact

For questions about this project, please contact:

- Liam Barrett - [l.barrett.16@ucl.ac.uk](mailto:l.barrett.16@ucl.ac.uk)

---

**Note**: This is an active research project. Code and documentation are being continuously updated as the manuscript undergoes review.
