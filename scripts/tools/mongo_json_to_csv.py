#!/usr/bin/env python3

"""
Process clinical annotation JSON files and organize them into CSV files.

This module processes two types of JSON files:
1. annotations.json - Contains clinical text annotations with qualifiers
2. snomedmappings.json - Contains SNOMED CT code mappings for annotations

The script organizes the data by username and letter ID, linking annotations with
their corresponding SNOMED codes. It outputs CSV files in the format:
./username/letter_id_annotations.csv

Each CSV includes annotation details (text, context, qualifiers) and any linked
SNOMED CT codes, with multiple codes concatenated using semicolons.

Usage:
   Simply run the script with both JSON files in the same directory.
   The resulting CSV files will be created in subdirectories named by username.

Author: Liam Barrett
Version: 1.0.0
Date: 04/03/2025
"""

import json
import os
from collections import defaultdict
import pandas as pd

# Step 1: Read the JSON files
with open('./nlp_ehr/results/mongodb/mongodb_export_20250530_145600/annotations.json', 'r') as f:
    annotations = json.load(f)

with open('./nlp_ehr/results/mongodb/mongodb_export_20250530_145600/snomedmappings.json', 'r') as f:
    snomed_mappings = json.load(f)

# Step 2: Create a dictionary to map annotation IDs to SNOMED codes
snomed_dict = {}
for mapping in snomed_mappings:
    annotation_id = mapping['annotationId']
    concept_ids = []
    for code in mapping['selectedCodes']:
        concept_ids.append(code['conceptId'])

    # Join multiple concept IDs with semicolons
    snomed_dict[annotation_id] = {
        'conceptIds': ';'.join(concept_ids),
        'timeSpent': mapping['timeSpent']
    }

# Step 3: Organize annotations by username and letter ID
user_letter_annotations = defaultdict(lambda: defaultdict(list))

for annotation in annotations:
    username = annotation['username']
    letter_id = annotation['letterId']

    # Extract required fields
    annotation_data = {
        "_id": annotation['_id'],
        "text": annotation['text'],
        "context": annotation['context'],
        "qualifier": annotation['qualifier'],
        "laterality": annotation['laterality'],
        "presence": annotation['presence'],
        "presentation": annotation['presentation'],
        "experiencer": annotation['experiencer'],
        "stage": annotation['stage'],
        "timeSpent": annotation['timeSpent']
    }

    # Add SNOMED codes if they exist
    if annotation['_id'] in snomed_dict:
        annotation_data['conceptIds'] = snomed_dict[annotation['_id']]['conceptIds']
        annotation_data['snomedTimeSpent'] = snomed_dict[annotation['_id']]['timeSpent']
    else:
        annotation_data['conceptIds'] = ""
        annotation_data['snomedTimeSpent'] = 0

    user_letter_annotations[username][letter_id].append(annotation_data)

# Step 4: Create directories and save CSV files
for username, letter_annotations in user_letter_annotations.items():
    # Create directory for the username
    os.makedirs(f'./{username}', exist_ok=True)

    for letter_id, annotations_list in letter_annotations.items():
        # Create dataframe
        df = pd.DataFrame(annotations_list)

        # Save to CSV
        output_path = f'./{username}/letter_{letter_id}_annotations.csv'
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")
