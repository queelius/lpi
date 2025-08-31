"""
Simple Linear Regression with Fisher Flow.

This example demonstrates:
1. How Fisher Flow provides both parameter estimates and uncertainty
2. Information accumulation from sequential data
3. Comparison with standard gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from fisherflow.autograd import Value, Module
from fisherflow.optimizers import DiagonalFisherFlow, AdamAsFisherFlow


class LinearRegression(Module):
    """Simple linear regression model: y = wx + b"""
    
    def __init__(self):
        self.w = Value(np.random.randn() * 0.1)
        self.b = Value(0.0)
    
    def __call__(self, x):
        return self.w * x + self.b
    
    def parameters(self):
        return [self.w, self.b]


def generate_data(n_samples=100, noise=0.1):
    """Generate synthetic linear data."""
    X = np.random.uniform(-2, 2, n_samples)
    true_w, true_b = 2.5, -1.0
    y = true_w * X + true_b + np.random.randn(n_samples) * noise
    return X, y, true_w, true_b


def mse_loss(batch, model):
    """Mean squared error loss."""
    X, y = batch
    total_loss = Value(0)
    
    for xi, yi in zip(X, y):
        xi_val = Value(xi)
        pred = model(xi_val)
        loss = (pred - yi) ** 2
        total_loss = total_loss + loss
    
    return total_loss / len(X)


def train_with_fisher_flow(model, X, y, optimizer, epochs=50, batch_size=10):
    """Train model using Fisher Flow optimizer."""
    n_samples = len(X)
    history = {'loss': [], 'w': [], 'b': [], 'w_uncertainty': [], 'b_uncertainty': []}
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss = mse_loss((batch_X, batch_y), model)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            epoch_loss += loss.data
            n_batches += 1
        
        # Record history
        history['loss'].append(epoch_loss / n_batches)
        history['w'].append(model.w.data)
        history['b'].append(model.b.data)
        
        # Get uncertainty estimates (for diagonal Fisher Flow)
        if hasattr(optimizer, 'fisher_diag'):
            w_var = 1.0 / (optimizer.fisher_diag[model.w] + 1e-6)
            b_var = 1.0 / (optimizer.fisher_diag[model.b] + 1e-6)
            history['w_uncertainty'].append(np.sqrt(w_var))
            history['b_uncertainty'].append(np.sqrt(b_var))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {history['loss'][-1]:.4f}, "
                  f"w = {model.w.data:.3f}, b = {model.b.data:.3f}")
    
    return history


def visualize_results(X, y, model, history, true_w, true_b):
    """Visualize training results and uncertainty."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Data and fitted line
    ax = axes[0, 0]
    ax.scatter(X, y, alpha=0.5, label='Data')
    X_line = np.linspace(X.min(), X.max(), 100)
    y_pred = model.w.data * X_line + model.b.data
    ax.plot(X_line, y_pred, 'r-', label=f'Fisher Flow: y={model.w.data:.2f}x+{model.b.data:.2f}')
    ax.plot(X_line, true_w * X_line + true_b, 'g--', 
            label=f'True: y={true_w}x+{true_b}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Linear Regression Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss over time
    ax = axes[0, 1]
    ax.plot(history['loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Parameter convergence
    ax = axes[0, 2]
    ax.plot(history['w'], label='w (slope)', color='blue')
    ax.plot(history['b'], label='b (intercept)', color='orange')
    ax.axhline(y=true_w, color='blue', linestyle='--', alpha=0.5, label='True w')
    ax.axhline(y=true_b, color='orange', linestyle='--', alpha=0.5, label='True b')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Parameter Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Parameter uncertainty over time (if available)
    if 'w_uncertainty' in history and history['w_uncertainty']:
        ax = axes[1, 0]
        epochs = range(len(history['w_uncertainty']))
        ax.fill_between(epochs,
                        [history['w'][i] - history['w_uncertainty'][i] for i in epochs],
                        [history['w'][i] + history['w_uncertainty'][i] for i in epochs],
                        alpha=0.3, color='blue', label='w ± σ')
        ax.plot(epochs, history['w'], color='blue', label='w estimate')
        ax.axhline(y=true_w, color='blue', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Slope (w)')
        ax.set_title('Parameter Uncertainty (Slope)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.fill_between(epochs,
                        [history['b'][i] - history['b_uncertainty'][i] for i in epochs],
                        [history['b'][i] + history['b_uncertainty'][i] for i in epochs],
                        alpha=0.3, color='orange', label='b ± σ')
        ax.plot(epochs, history['b'], color='orange', label='b estimate')
        ax.axhline(y=true_b, color='orange', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Intercept (b)')
        ax.set_title('Parameter Uncertainty (Intercept)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Information accumulation
    ax = axes[1, 2]
    if 'w_uncertainty' in history and history['w_uncertainty']:
        # Fisher information is inverse of variance
        fisher_w = [1.0 / (u**2 + 1e-6) for u in history['w_uncertainty']]
        fisher_b = [1.0 / (u**2 + 1e-6) for u in history['b_uncertainty']]
        ax.plot(fisher_w, label='Fisher Info (w)', color='blue')
        ax.plot(fisher_b, label='Fisher Info (b)', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Fisher Information')
        ax.set_title('Information Accumulation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()


def compare_optimizers():
    """Compare different Fisher Flow variants."""
    # Generate data
    X, y, true_w, true_b = generate_data(n_samples=200, noise=0.2)
    
    print("=" * 60)
    print("Fisher Flow for Linear Regression")
    print("=" * 60)
    print(f"True parameters: w = {true_w}, b = {true_b}")
    print()
    
    # Train with Diagonal Fisher Flow
    print("Training with Diagonal Fisher Flow:")
    print("-" * 40)
    model_diag = LinearRegression()
    optimizer_diag = DiagonalFisherFlow(model_diag.parameters(), lr=0.1)
    history_diag = train_with_fisher_flow(model_diag, X, y, optimizer_diag, epochs=50)
    
    print("\nTraining with Adam (as Fisher Flow):")
    print("-" * 40)
    model_adam = LinearRegression()
    optimizer_adam = AdamAsFisherFlow(model_adam.parameters(), lr=0.01)
    history_adam = train_with_fisher_flow(model_adam, X, y, optimizer_adam, epochs=50)
    
    # Visualize results
    print("\nVisualizing Diagonal Fisher Flow results...")
    visualize_results(X, y, model_diag, history_diag, true_w, true_b)
    
    # Final comparison
    print("\n" + "=" * 60)
    print("Final Results Comparison:")
    print("-" * 60)
    print(f"True parameters:          w = {true_w:.3f}, b = {true_b:.3f}")
    print(f"Diagonal Fisher Flow:     w = {model_diag.w.data:.3f}, b = {model_diag.b.data:.3f}")
    print(f"Adam (as Fisher Flow):    w = {model_adam.w.data:.3f}, b = {model_adam.b.data:.3f}")
    
    if history_diag['w_uncertainty']:
        print(f"\nDiagonal FF Uncertainty:  σ_w = {history_diag['w_uncertainty'][-1]:.4f}, "
              f"σ_b = {history_diag['b_uncertainty'][-1]:.4f}")


if __name__ == "__main__":
    compare_optimizers()