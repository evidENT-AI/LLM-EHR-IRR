#!/usr/bin/env python3

"""
Single letter analysis script using Meta's LLAMA 3.1 via Google Vertex AI.

This script processes a single clinical letter using the LLAMA 3.1 model
for annotation and analysis.

Note you may need to run:

```bash
gcloud auth application-default login
```

In the local environment to authenticate with Google Cloud.

Author: Liam Barrett (adapted for LLAMA 3.1)
Version: 1.0.0
"""

import json
import vertexai
from google.auth import default, transport

from labeller import LlamaInstance
from config import (
    PROJECT_ID,
    LOCATION,
    LLAMA_MODELS
)

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Get credentials for Vertex AI
credentials, _ = default()
auth_request = transport.requests.Request()
credentials.refresh(auth_request)

# Initialize with output management
llama = LlamaInstance(
    project_id=PROJECT_ID,
    location=LOCATION,
    model=LLAMA_MODELS['LLAMA_8B'],
    temperature=0,
    output_dir="../../results/annotations/llm/meta/llama3-1-8b/",
    letter_id="letter_0041"
)

# Run analysis
llama.reset(letter_id="letter_0041")
try:
    # load letter
    PREPROCESSED_LETTER = llama.load_letter('../../data/mt_samples/letters/letter_0041.json')
    # run full annotation
    run_data = llama.full_letter_annotation(PREPROCESSED_LETTER)
except FileNotFoundError as e:
    print(f"Letter file not found: {str(e)}")
except json.JSONDecodeError as e:
    print(f"Error parsing letter JSON: {str(e)}")
except ValueError as e:
    print(f"Value error in processing: {str(e)}")
except IOError as e:
    print(f"IO operation failed: {str(e)}")
except KeyError as e:
    print(f"Key error in data processing: {str(e)}")
except Exception as e:  # Catch any unexpected errors
    print(f"Unexpected error occurred: {str(e)}")
    raise  # Re-raise unexpected exceptions for debugging