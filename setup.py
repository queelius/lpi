"""Setup script for Fisher Flow package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fisher-flow",
    version="0.1.0",
    author="Alex Towell",
    author_email="atowell@siue.edu",
    description="Information-geometric sequential inference framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fisher-flow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "nbsphinx>=0.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "fisher-flow-demo=examples.simple_demo:main",
        ],
    },
)