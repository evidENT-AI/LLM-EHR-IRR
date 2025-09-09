# Instructions for annotating letters

Your task is to extract and analyze information from clinical letters systematically and accurately.

Key Responsibilities:

1. Extract relevant clinical information into structured categories:
   - Socio-demographics
   - Signs
   - Symptoms
   - Diagnoses
   - Treatments
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
  - Key Text: The specific phrase or term from the source text
  - Context: The complete relevant sentence or passage containing the key text
- Optional Fields (include when applicable):
  - Laterality (left, right, bilateral, NA)
  - Presence (confirmed, suspected, resolved, NA)
  - Primary/Secondary classification
  - Experiencer (patient, family member, NA)
  - Treatment stage (pre-treatment, post-treatment, NA)
  - SNOMED-CT code (if confident in the mapping)

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

Below is an example format for the table:

| text | context | qualifier | laterality | presence | primary_secondary | experiencer | treatment_stage | snomed_ct |
|---|---|---|---|---|---|---|---|---|
| collar tubes technique | description: collar tubes technique | treatments |  |  | primary | patient |  |  |
| general anesthesia | the patient was mask induced under general anesthesia. | treatments |  | confirmed | primary | patient | pre-treatment |  |
| cerumen removal | under microotoscopy, cerumen was removed from both ear canals. | treatments | bilateral | confirmed | primary | patient | post-treatment | 431855005 |
| myringotomy incision | an incision was placed in the inferior portion of the tympanic membrane bilaterally. | treatments | bilateral | confirmed | primary | patient | post-treatment | 9E+17 |
| baron's suction | a baron's suction was used to aspirate through the myringotomy incision. | treatments |  | confirmed | primary | patient | post-treatment |  |
| collar button pe tube | a collar button pe tube was placed in both myringotomy sites. | treatments | bilateral | confirmed | primary | patient | post-treatment |  |
| ototopical drops | ototopical drops were then placed through the lumen of the pe tubes. | treatments |  | confirmed | primary | patient | post-treatment |  |
| stable condition | the patient was then awakened from general anesthesia and taken to the recovery room in stable condition. | signs |  | confirmed | primary | patient | post-treatment |  |