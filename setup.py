from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ablangpdb1",
    version="1.0.0",
    author="Clinton M. Holt",
    author_email="clinton.m.holt@vanderbilt.edu",
    description="Epitope-aware antibody embeddings for targeted therapeutic discovery",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/AbLangPDB1",
    project_urls={
        "Paper": "https://doi.org/10.1101/2025.02.25.640114",
        "HuggingFace Model": "https://huggingface.co/clint-holt/AbLangPDB1",
        "Bug Reports": "https://github.com/your-username/AbLangPDB1/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "benchmarking": [
            "scikit-learn>=1.0.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "openpyxl>=3.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
        "benchmarking": ["*.parquet", "*.py"],
    },
    entry_points={
        "console_scripts": [
            "ablangpdb1-quick-start=quick_start_example:main",
        ],
    },
    keywords=[
        "antibody", 
        "epitope", 
        "embedding", 
        "machine learning", 
        "bioinformatics", 
        "transformers",
        "protein language model",
        "therapeutic discovery",
        "contrastive learning"
    ],
    zip_safe=False,
)