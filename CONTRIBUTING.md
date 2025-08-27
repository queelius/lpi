# Contributing to Fisher Flow

Thank you for your interest in contributing to Fisher Flow! This project aims to develop and explore information-geometric methods for sequential inference.

## Ways to Contribute

### 1. Implement New Fisher Flow Variants
- Sparse Fisher Flow for high-dimensional models
- Graph-structured Fisher Flow for GNNs
- Mixture Fisher Flow for multi-modal distributions
- Quantum Fisher Flow for quantum machine learning

### 2. Add Applications
- Computer vision examples
- NLP applications with uncertainty
- Reinforcement learning with Fisher Flow
- Scientific computing applications

### 3. Improve Theory
- Prove tighter bounds
- Extend to non-regular models
- Connect to other theoretical frameworks
- Develop new approximation schemes

### 4. Enhance Implementation
- GPU acceleration with PyTorch/JAX
- Distributed Fisher Flow
- Automatic approximation selection
- Memory-efficient variants

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/fisher-flow.git
cd fisher-flow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## Code Style

We follow PEP 8 with a few modifications:
- Line length: 88 characters (Black default)
- Use type hints where helpful
- Document all public functions

Format code with:
```bash
black fisher_flow.py
isort fisher_flow.py
```

## Pull Request Process

1. **Fork and Branch**: Create a feature branch from `main`
2. **Implement**: Make your changes with clear commits
3. **Test**: Add tests for new functionality
4. **Document**: Update docstrings and README if needed
5. **PR**: Submit with clear description of changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Theory improvement
- [ ] Documentation

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Examples run successfully

## Notes
Any additional context
```

## Research Ideas to Explore

### High Priority
1. **Adaptive Structure Learning**: Automatically choose between diagonal, block, and full Fisher Flow based on data
2. **Non-Gaussian Extensions**: Beyond quadratic approximations
3. **Causal Fisher Flow**: Information geometry for causal inference

### Experimental
1. **Hyperbolic Fisher Flow**: For hierarchical data
2. **Topological Fisher Flow**: Incorporate topological information
3. **Meta Fisher Flow**: Learn to learn with information geometry

## Documentation

When adding new features:
1. Add docstrings with examples
2. Update relevant README sections
3. Add to tutorial notebook if appropriate
4. Consider writing a blog post for major features

## Community

- **Discussions**: Use GitHub Discussions for ideas and questions
- **Issues**: Report bugs or request features via GitHub Issues
- **Contact**: Reach out to atowell@siue.edu for research collaborations

## Citation

If your contribution leads to a publication, please cite the original Fisher Flow paper and acknowledge contributors appropriately.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them contribute
- Focus on constructive criticism
- Celebrate diverse perspectives

## License

By contributing, you agree that your contributions will be licensed under the MIT License.