"""Setup script for nonconvexAG package."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="nonconvexAG",
    version="0.9.6",
    author="Kai Yang",
    author_email="kai.yang2@mail.mcgill.ca",
    description="Accelerated gradient methods for nonconvex sparse learning with SCAD and MCP penalties",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kaiyangshi-Ito/nonconvexAG",
    project_urls={
        "Bug Tracker": "https://github.com/Kaiyangshi-Ito/nonconvexAG/issues",
        "Documentation": "https://github.com/Kaiyangshi-Ito/nonconvexAG",
        "Source Code": "https://github.com/Kaiyangshi-Ito/nonconvexAG",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.16.6",
        "numba>=0.54.1", 
        "scipy>=1.0.0",
        "multiprocess>=0.70.6",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "examples": [
            "matplotlib>=3.0",
            "jupyter>=1.0",
            "scikit-learn>=0.24",
        ],
    },
    keywords=[
        "sparse learning",
        "nonconvex optimization", 
        "SCAD penalty",
        "MCP penalty",
        "accelerated gradient",
        "high-dimensional statistics",
        "machine learning",
    ],
)