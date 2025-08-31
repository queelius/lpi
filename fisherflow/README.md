# Fisher Flow: Information-Geometric Framework for Sequential Estimation

A Python implementation of Fisher Flow (FF), demonstrating how many modern optimization methods (Adam, Natural Gradient, Elastic Weight Consolidation) are special cases of propagating Fisher information rather than probability distributions.

## Overview

Fisher Flow is a unified framework that:
- **Propagates Fisher Information** instead of full probability distributions
- **Achieves Cramér-Rao efficiency** asymptotically
- **Unifies modern optimizers** under information-geometric principles
- **Provides uncertainty quantification** with computational efficiency

### Key Insight

Instead of tracking all possible parameter values and their probabilities (expensive!), Fisher Flow tracks just two things:
1. **Current best estimate** (θ̂)
2. **Fisher Information Matrix** (I) - encoding uncertainty

When new data arrives, both are updated using simple matrix operations - no complex integration required!

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fisher-flow.git
cd fisher-flow/fisherflow

# Install the package
pip install -e .
```

### Dependencies
- `numpy>=1.19.0` - Numerical computations
- `scipy>=1.5.0` - Statistical functions
- `matplotlib>=3.2.0` - Visualization
- `micrograd>=0.1.0` - Automatic differentiation base

## Quick Start

### Simple Linear Regression with Uncertainty

```python
from fisherflow.autograd import Value, Module
from fisherflow.optimizers import DiagonalFisherFlow

# Define a simple linear model
class LinearModel(Module):
    def __init__(self):
        self.w = Value(0.1)
        self.b = Value(0.0)
    
    def __call__(self, x):
        return self.w * x + self.b
    
    def parameters(self):
        return [self.w, self.b]

# Create model and optimizer
model = LinearModel()
optimizer = DiagonalFisherFlow(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # ... compute loss ...
    loss.backward()
    optimizer.step()
    
    # Get parameter uncertainty
    w_uncertainty = 1.0 / np.sqrt(optimizer.fisher_diag[model.w])
    print(f"w = {model.w.data:.3f} ± {w_uncertainty:.3f}")
```

## Core Concepts

### 1. Fisher Information as Currency

Fisher Flow treats **information as the natural currency of learning**:
- Data provides information about parameters
- Information accumulates additively (I_total = I_1 + I_2)
- Confidence is inverse variance (more information = less uncertainty)

### 2. The Fisher Flow Update

The core update equations:
```
I_t = I_{t-1} + I_batch         # Information accumulation
θ_t = θ_{t-1} - η * I_t^{-1} * ∇L   # Natural gradient step
```

### 3. Approximation Hierarchy

Different computational budgets lead to different approximations:

| Approximation | Structure | Computational Cost | Use Case |
|--------------|-----------|-------------------|----------|
| Full Fisher Flow | Full I ∈ ℝ^(d×d) | O(d³) | Small models, high accuracy |
| Kronecker-Factored | I ≈ A ⊗ B | O(m³ + n³) | Neural network layers |
| Diagonal (Adam-like) | I = diag(v) | O(d) | Large models, fast training |
| Scalar (SGD) | I = λI | O(1) | Simple, baseline |

## How Popular Optimizers are Fisher Flow

### Adam = Diagonal Fisher Flow
```python
# Adam is diagonal Fisher Flow with:
# - First moment (momentum) as exponential moving average of gradients
# - Second moment as diagonal Fisher Information estimate

optimizer = AdamAsFisherFlow(params, lr=0.001, beta1=0.9, beta2=0.999)
```

### Natural Gradient = Full Fisher Flow
```python
# Natural gradient uses the full Fisher Information Matrix

optimizer = NaturalGradientFlow(params, lr=0.01)
optimizer.compute_fisher_matrix(model, data_loader, loss_fn)
```

### EWC = Fisher Flow Regularization
```python
# Elastic Weight Consolidation uses Fisher Information 
# to prevent catastrophic forgetting

optimizer = ElasticWeightConsolidation(params, lr=0.01, ewc_lambda=1000)
optimizer.consolidate(model, task_data, loss_fn)  # After each task
```

## Examples

### 1. Linear Regression with Uncertainty
```bash
cd examples
python simple_regression.py
```
Demonstrates:
- Parameter estimation with uncertainty quantification
- Information accumulation over time
- Comparison of different Fisher Flow variants

### 2. Continual Learning without Forgetting
```bash
python continual_learning.py
```
Shows how EWC (Fisher Flow regularization) prevents catastrophic forgetting when learning multiple tasks sequentially.

## Package Structure

```
fisherflow/
├── fisherflow/
│   ├── __init__.py          # Package initialization
│   ├── core.py              # Core Fisher Flow abstractions
│   ├── autograd.py          # Micrograd-style automatic differentiation
│   ├── optimizers.py        # Fisher Flow optimizer variants
│   └── utils.py             # Utility functions
├── examples/
│   ├── simple_regression.py # Linear regression example
│   └── continual_learning.py # EWC demonstration
├── tests/
│   └── test_core.py         # Unit tests
├── setup.py                 # Package setup
└── README.md               # This file
```

## Mathematical Foundation

Fisher Flow is grounded in **information geometry** - viewing parametric families as Riemannian manifolds with the Fisher-Rao metric.

### Key Properties

1. **Information Additivity**: For independent observations x₁, ..., xₙ:
   ```
   I_{1:n}(θ) = Σᵢ I_{xᵢ}(θ)
   ```

2. **Cramér-Rao Bound**: Fisher Flow achieves the theoretical limit of estimation efficiency:
   ```
   Var(θ̂) ≥ I⁻¹(θ)
   ```

3. **Geometric Invariance**: Updates are covariant under reparameterization

## Testing

Run the test suite:
```bash
cd tests
python -m pytest test_core.py -v
```

## Citation

If you use Fisher Flow in your research, please cite:

```bibtex
@article{towell2025fisherflow,
  title={Fisher Flow: An Information-Geometric Framework for Sequential Estimation},
  author={Towell, Alex},
  year={2025}
}
```

## Related Work

Fisher Flow builds on and unifies:
- **Information Geometry** (Amari, 1998)
- **Natural Gradient Descent** (Amari, 1998)
- **Adam Optimizer** (Kingma & Ba, 2014)
- **K-FAC** (Martens & Grosse, 2015)
- **Elastic Weight Consolidation** (Kirkpatrick et al., 2017)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This implementation is based on the Fisher Flow paper and demonstrates the unifying principle that many successful optimization methods are approximations of propagating Fisher information on statistical manifolds.