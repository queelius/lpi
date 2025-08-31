"""Setup script for Fisher Flow package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fisherflow",
    version="0.1.0",
    author="Fisher Flow Contributors",
    description="Information-Geometric Framework for Sequential Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fisherflow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "micrograd>=0.1.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
)