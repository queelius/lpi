# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Fisher Flow is a unified framework for information-geometric sequential inference that reveals how many optimization methods (Adam, Natural Gradient, K-FAC, EWC) are special cases of propagating Fisher information. The repository contains both the academic paper and Python implementation.

## Project Structure

### Documentation
- `paper-draft.tex` - Main LaTeX document for the academic paper
- `paper-draft.pdf` - Compiled PDF output
- `references.bib` - Bibliography with 33 academic references
- `README.md` - Comprehensive project documentation with examples

### Implementation
- `fisher_flow.py` - Core library with Fisher Flow variants:
  - `ScalarFF` - Single learning rate (SGD-like)
  - `DiagonalFF` - Per-parameter learning rates (Adam-like)
  - `FullFF` - Complete information matrix (Natural Gradient)
  - `BlockFF` - Block-diagonal structure
  - `KroneckerFF` - Kronecker-factored (K-FAC-like)
  - `ExponentialFF` - With forgetting for non-stationary data
- `setup.py` - Package configuration with dependencies
- `examples/simple_demo.py` - Demonstration of online linear regression

## Common Commands

### Python Development
```bash
# Install in development mode
pip install -e .

# Install with all extras (dev tools, visualization, docs)
pip install -e ".[dev,viz,docs]"

# Run the demo
python examples/simple_demo.py
# or via entry point
fisher-flow-demo

# Code quality (if dev extras installed)
black fisher_flow.py examples/     # Format code
isort fisher_flow.py examples/     # Sort imports
flake8 fisher_flow.py examples/    # Lint
mypy fisher_flow.py                # Type check

# Run the main module demo
python fisher_flow.py
```

### LaTeX Compilation
```bash
# Full compilation with bibliography
pdflatex paper-draft.tex
bibtex paper-draft
pdflatex paper-draft.tex
pdflatex paper-draft.tex

# Quick compilation (without bibliography)
pdflatex paper-draft.tex

# Clean auxiliary files
rm -f *.aux *.log *.bbl *.blg *.out *.toc *.synctex.gz
```

## Architecture & Design

### Core Abstraction
All Fisher Flow variants inherit from `FisherFlow` base class and implement:
- `_init_information()` - Initialize information matrix structure
- `_update_information()` - Accumulate new gradient/Hessian information
- `_get_covariance()` - Return uncertainty (inverse information)
- `_compute_natural_gradient()` - Apply information geometry

### Key Methods
- `update(grad, hess, learning_rate)` - Main update step
- `confidence_interval(alpha)` - Parameter uncertainty bounds
- `predict(X, link_function)` - Predictions with uncertainty

### Information Accumulation Pattern
```python
# Core update equation
I_t = I_{t-1} + F(batch_t)  # Accumulate Fisher information
θ_t = I_t^{-1} (I_{t-1} θ_{t-1} + F(batch_t) θ_batch)  # Natural gradient step
```

## Key Technical Concepts

- **Fisher Information Matrix**: Curvature of log-likelihood, quantifies information about parameters
- **Natural Gradient**: Parameter updates that respect information geometry
- **Sequential Inference**: Online learning with uncertainty quantification
- **Computational Trade-offs**: Different approximations (diagonal, block, Kronecker) balance accuracy vs efficiency