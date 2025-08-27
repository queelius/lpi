# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an academic paper and implementation repository for "Fisher Flow" - a unified framework for information-geometric sequential inference that shows many optimization methods (Adam, Natural Gradient, EWC) are special cases of propagating Fisher information.

## Project Structure

- `paper-draft.tex` - Main LaTeX document containing the full paper
- `paper-draft.pdf` - Compiled PDF output
- `references.bib` - Bibliography with 33 academic references

## Common Commands

### Compile LaTeX document
```bash
# Full compilation with bibliography
pdflatex paper-draft.tex
bibtex paper-draft
pdflatex paper-draft.tex
pdflatex paper-draft.tex

# Quick compilation (without bibliography update)
pdflatex paper-draft.tex
```

### Clean auxiliary files
```bash
rm -f *.aux *.log *.bbl *.blg *.out *.toc *.synctex.gz
```

## Paper Architecture & Content

The paper presents Fisher Flow (FF), a framework that:
- Provides an alternative to Bayesian sequential updating using maximum likelihood estimation
- Leverages the Fisher Information Matrix for uncertainty quantification
- Connects to modern deep learning optimization methods (Adam, RMSProp, natural gradients)
- Relates to continual learning techniques (Elastic Weight Consolidation)

### Document Structure
The LaTeX document follows standard academic paper organization:
1. Abstract and Introduction - Motivation and overview
2. Mathematical Framework - Core LPI theory and Fisher Information
3. Connections to Existing Methods - Links to ML optimization techniques
4. Applications - Deep learning, signal processing, control systems
5. Discussion and Conclusion - Implications and future work

### Key Technical Concepts
When editing, be aware of the paper's focus on:
- Information geometry and Fisher Information Matrix
- Recursive maximum likelihood updates
- Connections between statistical inference and deep learning optimization
- Computational efficiency vs. full Bayesian inference