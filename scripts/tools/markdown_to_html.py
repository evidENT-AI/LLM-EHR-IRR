#!/usr/bin/env python3
"""Markdown to HTML/PDF Converter.

This module converts markdown files to HTML with consistent styling,
suitable for both web viewing and PDF conversion.

Author: Liam Barrett
Date: December 2024
"""

import argparse
from pathlib import Path
import re
import subprocess
from typing import Optional

import markdown
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.toc import TocExtension
from markdown.extensions.codehilite import CodeHiliteExtension

class MarkdownConversionError(Exception):
    """Custom exception for markdown conversion errors."""
    pass

def clean_text(text: str) -> str:
    """Clean text by removing special characters and normalizing whitespace.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text string
    """
    # Remove special characters but keep necessary markdown syntax
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    # Normalize whitespace but preserve markdown line breaks
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    return text.strip()

def load_css_file(css_path: str = "styles.css") -> str:
    """Load the CSS file from disk.
    
    Args:
        css_path: Path to the CSS file.
        
    Returns:
        The contents of the CSS file as a string.
        
    Raises:
        FileNotFoundError: If the CSS file cannot be found.
    """
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"CSS file not found at {css_path}. Please ensure 'styles.css' is in the same directory."
        ) from exc

def extract_title_from_markdown(markdown_content: str) -> str:
    """Extract title from markdown content.
    
    Looks for the first h1 header (# Title) in the markdown content.
    
    Args:
        markdown_content: The markdown content as string.
        
    Returns:
        The title string, or 'Untitled Document' if no title found.
    """
    # Clean the content first
    clean_content = clean_text(markdown_content)
    
    # Look for # Title pattern first (most common)
    h1_pattern = r'^\s*#\s+([^\n]+)'
    match = re.search(h1_pattern, clean_content, re.MULTILINE)
    if match:
        return clean_text(match.group(1))
    
    # Look for Title\n=== pattern
    underline_pattern = r'^([^\n]+)\n=+\s*$'
    match = re.search(underline_pattern, clean_content, re.MULTILINE)
    if match:
        return clean_text(match.group(1))
    
    return "Untitled Document"

def create_html_document(title: str, content: str, style: str) -> str:
    """Create a complete HTML document.
    
    Args:
        title: Document title
        content: HTML content
        style: CSS styles
        
    Returns:
        Complete HTML document as string
    """
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
    {content}
</body>
</html>"""

def convert_markdown_to_html(
    input_path: str,
    output_path: Optional[str] = None,
    css_path: str = "styles.css"
) -> str:
    """Convert markdown file to HTML with styling.
    
    Args:
        input_path: Path to input markdown file
        output_path: Optional path for output HTML file
        css_path: Path to CSS file
        
    Returns:
        Path to the created HTML file
        
    Raises:
        FileNotFoundError: If input files cannot be found
        PermissionError: If there are permission issues
        MarkdownConversionError: For other conversion errors
    """
    try:
        # Read markdown content
        with open(input_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Clean the content
        md_content = clean_text(md_content)
        
        # Extract title
        title = extract_title_from_markdown(md_content)
        
        # Load CSS
        style = load_css_file(css_path)
        
        # Configure Markdown converter
        md = markdown.Markdown(extensions=[
            'markdown.extensions.extra',  # Tables, attrs, footnotes, etc.
            'markdown.extensions.meta',   # Metadata
            'markdown.extensions.smarty', # Smart quotes
            'markdown.extensions.sane_lists',
            FencedCodeExtension(),
            TableExtension(),
            TocExtension(permalink=True),
            CodeHiliteExtension(css_class='highlight')
        ])
        
        # Convert markdown to HTML
        html_content = md.convert(md_content)
        
        # Create complete HTML document
        html_doc = create_html_document(title, html_content, style)
        
        # Determine output path
        if output_path is None:
            output_path = str(Path(input_path).with_suffix('.html'))
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_doc)
        
        return output_path
        
    except (FileNotFoundError, PermissionError) as exc:
        raise exc
    except Exception as exc:
        raise MarkdownConversionError(f"Error converting markdown: {str(exc)}") from exc

def convert_to_pdf(html_path: str, pdf_path: Optional[str] = None) -> str:
    """Convert HTML file to PDF using wkhtmltopdf.
    
    Args:
        html_path: Path to input HTML file
        pdf_path: Optional path for output PDF file
        
    Returns:
        Path to the created PDF file
        
    Raises:
        FileNotFoundError: If input file cannot be found
        subprocess.CalledProcessError: If PDF conversion fails
    """
    if pdf_path is None:
        pdf_path = str(Path(html_path).with_suffix('.pdf'))
    
    try:
        subprocess.run([
            'wkhtmltopdf',
            '--enable-local-file-access',
            '--print-media-type',
            '--margin-top', '4cm',
            '--margin-bottom', '4cm',
            '--margin-left', '2.5cm',
            '--margin-right', '2.5cm',
            html_path,
            pdf_path
        ], check=True, capture_output=True, text=True)
        
        return pdf_path
        
    except subprocess.CalledProcessError as exc:
        raise MarkdownConversionError(f"PDF conversion failed: {exc.stderr}") from exc

def main():
    """Main function to run the conversion process."""
    parser = argparse.ArgumentParser(description='Convert markdown to HTML and optionally PDF')
    parser.add_argument('input', help='Input markdown file')
    parser.add_argument('--html', help='Output HTML file')
    parser.add_argument('--pdf', help='Output PDF file')
    parser.add_argument('--css', default='styles.css', help='CSS file path')
    
    args = parser.parse_args()
    
    try:
        # Convert to HTML
        html_path = convert_markdown_to_html(args.input, args.html, args.css)
        print(f"Created HTML file: {html_path}")
        
        # Convert to PDF if requested
        if args.pdf:
            pdf_path = convert_to_pdf(html_path, args.pdf)
            print(f"Created PDF file: {pdf_path}")
            
        return 0
        
    except (FileNotFoundError, PermissionError, MarkdownConversionError) as exc:
        print(f"Error: {exc}")
        return 1

if __name__ == "__main__":
    exit(main())