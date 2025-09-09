#!/usr/bin/env python3
"""Clinical Letter Processor.

This module processes clinical letters from JSON format to HTML and PDF formats.
It uses wkhtmltopdf for PDF conversion and applies consistent styling across all documents.

Author: Liam Barrett
Date: 09/12/2024
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Tuple

def load_css_file(css_path: str = "styles.css") -> str:
    """Load the CSS file from disk."""
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "CSS file not found. Please ensure 'styles.css' is in the same directory."
        ) from exc

def create_html_template(title: str, content: str, style: str) -> str:
    """Create HTML document with style and content."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
    {style}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {content}
</body>
</html>"""

def format_section(key: str, value: str) -> str:
    """Format a section of the clinical letter."""
    # Handle lists (like diagnoses)
    if any(str(i) + "." in value[:10] for i in range(1, 10)):
        items = [item.strip() for item in value.split("   ") if item.strip()]
        formatted_items = "".join([f"<li>{item}</li>" for item in items])
        return f"""<h2>{key}</h2>
<ul>
    {formatted_items}
</ul>"""

    return f"""<h2>{key}</h2>
<p>{value}</p>"""

def process_clinical_letter(letter_json: Dict) -> str:
    """Process clinical letter JSON into formatted HTML sections."""
    exclude_keys = {'url', 'medical specialty', 'sample name', 'keywords'}
    sections = []

    # Add description first if it exists
    if 'description' in letter_json:
        sections.append(format_section('Description', letter_json['description']))
        exclude_keys.add('description')

    # Process remaining sections
    for key, value in letter_json.items():
        if key not in exclude_keys and value and str(value).strip():
            display_title = key.replace('_', ' ').title()
            sections.append(format_section(display_title, value))

    return "\n".join(sections)

def create_output_directories(base_dir: str) -> Tuple[Path, Path]:
    """Create and return paths for HTML and PDF output directories."""
    html_dir = Path(base_dir) / 'letters_html'
    pdf_dir = Path(base_dir) / 'letters_pdf'

    html_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    return html_dir, pdf_dir

def json_to_outputs(input_dir: str, output_dir: str, css_path: str = "styles.css"):
    """Convert all JSON files in a directory to both HTML and PDF."""
    # Create output directories
    html_dir, pdf_dir = create_output_directories(output_dir)

    # Load external CSS file
    try:
        style = load_css_file(css_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Process each JSON file
    for json_file in Path(input_dir).glob('*.json'):
        try:
            # Load JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                letter_data = json.load(f)

            # Generate title from sample name or filename
            title = letter_data.get('sample name', json_file.stem)

            # Process letter content
            content = process_clinical_letter(letter_data)

            # Create HTML
            html = create_html_template(title, content, style)

            # Save HTML file
            html_path = html_dir / f"{json_file.stem}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)

            # Create PDF
            pdf_path = pdf_dir / f"{json_file.stem}.pdf"
            subprocess.run([
                'wkhtmltopdf',
                '--enable-local-file-access',
                '--print-media-type',
                '--margin-top', '4cm',
                '--margin-bottom', '4cm',
                '--margin-left', '2.5cm',
                '--margin-right', '2.5cm',
                str(html_path),
                str(pdf_path)
            ], check=True)

            print(f"Successfully processed {json_file.name} to HTML and PDF")

        except PermissionError as exc:
            raise PermissionError(
                f"Permission denied when processing {json_file.name}"
            ) from exc

def main():
    """Main function to run the conversion process."""
    # Define directories
    project_root = "Users/liambarrett/Evident-AI/nlp_ehr/"
    input_dir = f"{project_root}data/mt_samples/letters"  # Directory containing JSON files
    output_dir = f"{project_root}data/mt_samples/formatted_letters"        # Base output directory
    css_path = f"{project_root}docs/styles.css"      # Path to the CSS file

    # Process all files
    json_to_outputs(input_dir, output_dir, css_path)

if __name__ == "__main__":
    main()
