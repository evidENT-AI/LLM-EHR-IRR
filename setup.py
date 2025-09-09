#!/usr/bin/env python
"""Setup script for NLP-EHR-IRR framework."""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="nlp-ehr-irr",
    version="1.0.0",
    author="Liam Barrett",
    author_email="l.barrett.16@ucl.ac.uk",
    description="LLM evaluation framework for clinical documentation analysis",
    url="https://github.com/yourusername/nlp-ehr-irr",
    packages=find_packages(include=["scripts", "scripts.*"]),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
        ],
    },
    include_package_data=True,
    package_data={
        "scripts": ["*.json", "*.css"],
    },
    zip_safe=False,
)
