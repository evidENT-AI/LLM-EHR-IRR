#!/usr/bin/env python3
"""
Module for managing prompts in the NLP-EHR project.
Contains templates and functions for generating prompts for LLM interactions.
"""

import json

# Step 1: pre-prompt
PREPROMPT = '''
You are an expert medical professional with extensive experience in clinical documentation analysis. Your task is to extract and analyze information from clinical letters systematically and accurately.

Key Responsibilities:
1. Extract relevant clinical information into structured categories:
   - Socio-demographics
   - Signs
   - Symptoms
   - Diagnoses
   - Treatments
   - Procedures
   - Risk Factors
   - Test Results

Standards for Analysis:
- Base all extractions on explicit evidence from the text
- Maintain clinical precision and accuracy
- Avoid inferring information not directly supported by the text
- Distinguish between confirmed and suspected findings
- Consider temporal relationships in the clinical narrative
- Preserve medical terminology as presented in the source text

Required Information for Each Extraction:
- Mandatory Fields:
  * Key Text: The specific phrase or term from the source text
  * Context: The complete relevant sentence or passage containing the key text
- Optional Fields (include when applicable):
  * Laterality (left, right, bilateral, NA)
  * Presence (confirmed, suspected, resolved, NA)
  * Primary/Secondary classification
  * Experiencer (patient, family member, NA)
  * Treatment stage (pre-treatment, current, post-treatment, NA)
  * SNOMED-CT code (if confident in the mapping)

Output Requirements:
- Maintain clinical accuracy and precision
- Present information in a structured, consistent format
- Use "NA" for any field where information is not applicable or cannot be determined from the text
- Preserve the original medical terminology

Remember:
- Only extract information explicitly present in the text
- Maintain medical accuracy and precision
- Follow a systematic approach to analysis
- Be prepared to justify each extraction with specific textual evidence
- Flag any uncertainties or ambiguities in the source text
'''

# Step 2: Provide the aims of the task and an example annotation
EXAMPLE_LETTER_PROMPT = '''
I will now show you an example clinical letter and its annotation to demonstrate the task. Your job will be to replicate this type of analysis on new clinical letters.

Example Clinical Letter (Letter 0080):
{
  "letter": 0080,
  "description": "The patient had tympanoplasty surgery for a traumatic perforation of the right ear about six weeks ago.",
  "history": "The patient had tympanoplasty surgery for a traumatic perforation of the right ear about six weeks ago...."
  [rest of letter content as shown above]
}

Here is an example of how this letter should be analyzed, with each extracted label and its associated metadata in JSON format:

{
    "labels": [
      {
        "text": "tympanoplasty",
        "context": "The patient had tympanoplasty surgery for a traumatic perforation of the right ear about six weeks ago.",
        "qualifier": "treatment",
        "laterality": "right",
        "presence": "Confirmed",
        "primary_secondary": "Primary",
        "experienced": "patient",
        "treatment_stage": "Post-treatment",
        "snomed_ct": "386556002"
      },
      {
        "text": "traumatic perforation",
        "context": "The patient had tympanoplasty surgery for a traumatic perforation of the right ear about six weeks ago.",
        "qualifier": "diagnoses",
        "laterality": "right",
        "presence": "Confirmed",
        "primary_secondary": "Primary",
        "experienced": "patient",
        "treatment_stage": "Pre-treatment",
        "snomed_ct": "43892002"
      },
      {
        "text": "bothering him",
        "context": "Mom called because his other ear was bothering him.",
        "qualifier": "symptoms",
        "laterality": "left",
        "presence": "Confirmed",
        "primary_secondary": "Primary",
        "experienced": "patient",
        "treatment_stage": "Post-treatment",
        "snomed_ct": "NA"
      },
      {
        "text": "ringing",
        "context": "Earlier today, he said it was ringing and vibrating",
        "qualifier": "symptoms",
        "laterality": "left",
        "presence": "Confirmed",
        "primary_secondary": "Primary",
        "experienced": "patient",
        "treatment_stage": "Post-treatment",
        "snomed_ct": "162352007"
      },
      {
        "text": "vibrating",
        "context": "Earlier today, he said it was ringing and vibrating",
        "qualifier": "symptoms",
        "laterality": "left",
        "presence": "Confirmed",
        "primary_secondary": "Primary",
        "experienced": "patient",
        "treatment_stage": "Post-treatment",
        "snomed_ct": "NA"
      }
    ]
}

For each new clinical letter, you should:
1. Extract all relevant labels that fall into these categories:
   - Socio-demographics
   - Signs
   - Symptoms
   - Diagnoses
   - Treatments
   - Procedures
   - Risk Factors
   - Test Results

2. For each label, provide:
   - The exact text from the letter
   - The complete context (full sentence or relevant passage)
   - Appropriate qualifier from the categories above
   - All applicable metadata fields (laterality, presence, etc.)
   - Use "NA" for any field that is not applicable or cannot be determined

3. Format your response as a JSON object following the exact structure shown in the example above.

4. Ensure that:
   - Each label is an exact quote from the text
   - Context provides sufficient information to understand the label
   - All fields are completed (using "NA" when not applicable)
   - The classification is based solely on information present in the text

Are you ready to analyze a new clinical letter following this format?
'''

# Step 3: Create in-context learning session
def create_feedback_prompt(json_path):
    """
    Creates a feedback prompt with properly formatted JSON example for in-context learning.
    
    Args:
        json_path (str): Path to the JSON file containing the annotated letter
        
    Returns:
        str: Formatted prompt template with embedded JSON
    """
    # Read and format the JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Convert the JSON to a formatted string with proper indentation
    formatted_json = json.dumps(annotations, indent=2)
    
    FEEDBACK_PROMPT = f'''
Well done, that was a good first attempt. Here is what an expert consultant created for the same letter for reference. Consider the differences between the example, gold-standard annotation (below) and your own. Consider the different qualifiers that you may have missed out on and the information that was in the letter that your missed in your annotations.

{formatted_json}

DO NOT re-attempt this annotation. Rather, learning the deep structural components that are contributed to this gold-standard annotation JSON. We will now progress on to the next letter. This will be completely independent of the previous letter and so, you should learn the deep, structural patterns behind good annotations rather than specific items within the previous letter.

Are you ready?
'''
    return FEEDBACK_PROMPT

# Step 3: Prompt the LLM with the letter of choice
formatted_letter = 'None' # Initialise empty letter
def create_letter_prompt(formatted_letter):
    LETTER_PROMPT_DESC = '''
Analyze the following clinical letter and extract all relevant information. Your response must be a valid JSON object matching the following specification exactly.

Required Format:
{
  "labels": {
    "exact_text": { # replace exact_text with the label. In the examples, one was "15-year-old".
      "context": "string with complete sentence/passage",
      "qualifier": "MUST BE ONE OF: Sociodemographics, Signs, Symptoms, Diagnoses, Treatments, Procedures, Risk Factors, Test Results",
      "laterality": "MUST BE ONE OF: left, right, bilateral, NA",
      "presence": "MUST BE ONE OF: confirmed, suspected, resolved, negated, NA",
      "primary_secondary": "MUST BE ONE OF: primary, secondary, NA",
      "experiencer": "MUST BE ONE OF: Patient, Family, NA",
      "treatment_stage": "MUST BE ONE OF: pre-treatment, post-treatment, NA",
      "snomed_ct": "string of numbers or NA"
    }
    "exact_text": { # the rest of the labels
      // ... same structure as above
    }
  }
}

Critical Requirements:
1. Response must be a single JSON object
2. Include all labels you identify
3. All field names must be exactly as shown above
4. Every label must have ALL fields specified (use "NA" if not applicable)
5. No additional fields or nested objects are allowed
6. All string values must be properly escaped
7. Fields must use ONLY the enumerated values where specified
8. Context must contain complete sentences, properly escaped
9. Do not include any explanatory text before or after the JSON

Extract all instances of:
- Socio-demographics
- Signs
- Symptoms
- Diagnoses
- Treatments
- Procedures
- Risk Factors
- Test Results

Clinical Letter:
'''
    LETTER_PROMPT = LETTER_PROMPT_DESC+f'''

{formatted_letter}

Respond only with the JSON object following the specified format.
'''
    return LETTER_PROMPT

# Step 4: Review the extracted JSON for errors
formatted_labels_json = 'None' # Initialise empty JSON
REVIEW_LABELS_PROMPT_INTRO = '''
Review the following extracted labels from a clinical letter. Verify their accuracy, completeness, and consistency. The current extraction is:

'''
REVIEW_LABELS_PROMPT_MAIN = '''
Review Tasks:
1. MISSING INFORMATION
   Check for any missing labels in these categories:
   - Socio-demographics
   - Signs
   - Symptoms
   - Diagnoses
   - Treatments
   - Procedures
   - Risk Factors
   - Test Results

2. VALIDATION
   For each existing label, verify:
   - Context matches the label
   - Qualifier category is appropriate
   - Presence status is accurate
   - Primary/Secondary classification is correct
   - Experiencer assignment is accurate
   - Treatment stage is appropriate

3. CLINICAL CONSISTENCY
   Check that:
   - Related symptoms/signs are consistently labeled
   - Temporal relationships are correctly reflected
   - Family history is properly distinguished
   - Diagnostic relationships are accurate

Provide your response in the following JSON format:
{
  "review_status": {
    "missing_labels": [
      {
        "text": "exact text from letter",
        "context": "complete sentence/passage",
        "qualifier": "one of the valid categories",
        "laterality": "left/right/bilateral/NA",
        "presence": "confirmed/suspected/resolved/negated/NA",
        "primary_secondary": "primary/secondary/NA",
        "experiencer": "Patient/Family/NA",
        "treatment_stage": "pre-treatment/current/post-treatment/NA",
        "snomed_ct": "code or NA"
      }
    ],
    "corrections": {
      "label_text": {
        "field_to_correct": "new_value",
        "reason": "brief explanation"
      }
    },
    "deletions": [
      {
        "label_text": "text to remove",
        "reason": "brief explanation"
      }
    ]
  }
}

If no changes are needed, respond with:
{
  "review_status": {
    "missing_labels": [],
    "corrections": {},
    "deletions": []
  }
}

Original Clinical Letter:
'''

def create_review_label_prompt(formatted_labels_json, formatted_letter):
    REVIEW_LABELS_PROMPT = REVIEW_LABELS_PROMPT_INTRO+f'''
{formatted_labels_json}
'''+REVIEW_LABELS_PROMPT_MAIN+f'''

{formatted_letter}

Provide only the JSON response without any additional text.
'''

    return REVIEW_LABELS_PROMPT

# Step 4: Get modification/additions
operation_type = 'None' # Initialise necessary variables
label_text = 'None'
reason = 'None'
current_values_json = 'None'
existing_labels_summary = 'None'

def create_modify_add_prompt(operation_type, label_text, reason, current_values_json, existing_labels_summary, formatted_letter):
    MODIFY_OR_ADD_LABEL_PROMPT = f'''
Update or add the following label based on the clinical letter. Generate a complete JSON entry following the same format as the original extraction.

Operation Type: {operation_type}  # "modify" or "add"
Label Text: {label_text}
Reason for Change: {reason}

For modification, current values are:
{current_values_json}

Clinical Letter Context:
{formatted_letter}

Provide a single JSON entry for this label using this exact format:'''+'''
{
  "text": "'''+f'{label_text}'+'''",
  "context": "complete sentence/passage containing the label",
  "qualifier": "MUST BE ONE OF: Sociodemographics, Signs, Symptoms, Diagnoses, Treatments, Procedures, Risk Factors, Test Results",
  "laterality": "MUST BE ONE OF: left, right, bilateral, NA",
  "presence": "MUST BE ONE OF: confirmed, suspected, resolved, negated, NA",
  "primary_secondary": "MUST BE ONE OF: primary, secondary, NA",
  "experiencer": "MUST BE ONE OF: Patient, Family, NA",
  "treatment_stage": "MUST BE ONE OF: pre-treatment, current, post-treatment, NA",
  "snomed_ct": "code or NA"
}

Requirements:
1. Use exact text from the letter
2. Provide complete context sentence
3. Only use specified valid values for each field
4. Use "NA" for any field that doesn't apply
5. Ensure consistency with other labels
6. Consider temporal and clinical relationships

Original Extraction Context:'''+f'''
{existing_labels_summary}

Provide only the JSON entry without any additional text or explanation.
'''
    return MODIFY_OR_ADD_LABEL_PROMPT

# SNOMED searcher prompt
SNOMED_CODE_SELECTION_PROMPT = '''
You are tasked with selecting the most appropriate SNOMED CT code for a clinical annotation.

Original Annotation:
Text: {text}
Context: {context}
Qualifier: {qualifier}

Retrieved SNOMED CT codes:
{snomed_options}

Instructions:
1. Review the original annotation and the retrieved SNOMED CT codes carefully
2. Consider both the term and context when selecting a code
3. Select the most appropriate code OR indicate none are suitable

Choose one of the following responses:
1. If an appropriate code exists: Return the code (e.g., "123456")
2. If no appropriate code exists: Return "NaN"

Please provide just the code or "NaN" with no additional explanation.'''

SNOMED_SEARCH_REFINEMENT_PROMPT = '''The term "{text}" returned no results when searching SNOMED CT. This could be because:
1. There is no SNOMED CT code for this concept
2. The wording is too specific or complex for the database search

Please consider "{text}" and it context:

"{context}"

Please suggest a simpler or more standard medical term to search for in SNOMED CT. Consider:
- Use standard medical terminology
- Break compound terms into their core concept
- Retains the appropriate clinical meaning given the context
- Ensure the suggested term maintains the clinical meaning

Provide only the suggested search term with no additional explanation.'''