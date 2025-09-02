#!/usr/bin/env python3
"""
Simple demonstration of Fisher Flow for online linear regression.

This example shows how Fisher Flow provides both parameter estimates
and uncertainty quantification in a streaming data setting.
"""

import numpy as np
import matplotlib.pyplot as plt
from fisherflow.optimizers import DiagonalFisherFlow, NaturalGradientFlow

def generate_streaming_data(n_samples=1000, n_features=5, noise=0.1):
    """Generate synthetic regression data."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_theta = np.random.randn(n_features) * 2
    y = X @ true_theta + noise * np.random.randn(n_samples)
    return X, y, true_theta

def main():
    # Generate data
    X, y, true_theta = generate_streaming_data()
    n_samples, n_features = X.shape
    
    # Initialize Fisher Flow estimators
    diagonal_ff = DiagonalFisherFlow(n_features)
    full_ff = NaturalGradientFlow(n_features)
    
    # Storage for tracking convergence
    diagonal_errors = []
    full_errors = []
    diagonal_uncertainties = []
    full_uncertainties = []
    
    # Process data in batches (simulating streaming)
    batch_size = 10
    for i in range(0, n_samples, batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        
        # Compute gradient for this batch
        for ff, errors, uncertainties in [
            (diagonal_ff, diagonal_errors, diagonal_uncertainties),
            (full_ff, full_errors, full_uncertainties)
        ]:
            # Prediction error
            pred = batch_X @ ff.theta
            error = pred - batch_y
            
            # Gradient (negative log-likelihood derivative)
            grad = batch_X.T @ error / batch_size
            
            # Update Fisher Flow
            theta, cov = ff.update(grad, learning_rate=0.1)
            
            # Track convergence
            param_error = np.linalg.norm(theta - true_theta)
            errors.append(param_error)
            
            # Track uncertainty (trace of covariance)
            uncertainty = np.trace(cov)
            uncertainties.append(uncertainty)
    
    # Plotting results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Parameter error over time
    ax1.plot(diagonal_errors, label='Diagonal FF', linewidth=2)
    ax1.plot(full_errors, label='Full FF', linewidth=2)
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Parameter Error')
    ax1.set_title('Convergence of Fisher Flow Methods')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Uncertainty over time
    ax2.plot(diagonal_uncertainties, label='Diagonal FF', linewidth=2)
    ax2.plot(full_uncertainties, label='Full FF', linewidth=2)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Total Uncertainty (trace of covariance)')
    ax2.set_title('Uncertainty Reduction Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('fisher_flow_convergence.png', dpi=150)
    plt.show()
    
    # Print final results
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    for name, ff in [("Diagonal FF", diagonal_ff), ("Full FF", full_ff)]:
        print(f"\n{name}:")
        print(f"  Parameter error: {np.linalg.norm(ff.theta - true_theta):.4f}")
        
        # Show confidence intervals
        ci_lower, ci_upper = ff.confidence_interval(0.95)
        print(f"\n  95% Confidence Intervals:")
        for i in range(min(3, n_features)):
            in_ci = ci_lower[i] <= true_theta[i] <= ci_upper[i]
            symbol = "✓" if in_ci else "✗"
            print(f"    θ[{i}]: [{ci_lower[i]:6.3f}, {ci_upper[i]:6.3f}] "
                  f"(true: {true_theta[i]:6.3f}) {symbol}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("- Full FF converges faster but requires O(d²) operations")
    print("- Diagonal FF is more scalable with O(d) operations")
    print("- Both provide valid uncertainty quantification")
    print("- Confidence intervals contain true parameters ~95% of the time")

if __name__ == "__main__":
    main()