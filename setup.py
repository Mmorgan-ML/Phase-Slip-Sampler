# setup.py

"""
 Original Author: Michael Christian Morgan
 2025.12.03
 Github: https://github.com/Mmorgan-ML
 Twitter: @Mmorgan_ML
 Email: mmorgankorea@gmail.com
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of your README file safely
this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')
else:
    long_description = "A dynamic inference sampler for LLMs."

setup(
    name="phase-slip-sampler",
    version="0.1.1",
    author="Michael Christian Morgan",
    author_email="mmorgankorea@gmail.com",  # Fixed the missing comma
    description="A dynamic inference sampler for LLMs using thermodynamic thermal shock.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mmorgan-ML/Phase-Slip-Sampler",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8', # Bumped to 3.8 for better typing support
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.0.0",
        "numpy",
    ],
)