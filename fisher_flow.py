"""
Fisher Flow: Information-Geometric Sequential Inference

A unified framework for propagating Fisher information rather than 
probability distributions for efficient sequential parameter estimation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union
import warnings


class FisherFlow(ABC):
    """Abstract base class for Fisher Flow variants."""
    
    def __init__(self, dim: int, regularization: float = 1e-6):
        """
        Initialize Fisher Flow estimator.
        
        Args:
            dim: Parameter dimension
            regularization: Small constant for numerical stability
        """
        self.dim = dim
        self.regularization = regularization
        self.reset()
    
    def reset(self):
        """Reset the estimator to initial state."""
        self.theta = np.zeros(self.dim)
        self.n_samples = 0
        self._init_information()
    
    @abstractmethod
    def _init_information(self):
        """Initialize the information matrix."""
        pass
    
    @abstractmethod
    def _update_information(self, grad: np.ndarray, hess: Optional[np.ndarray] = None):
        """Update the information matrix with new data."""
        pass
    
    @abstractmethod
    def _get_covariance(self) -> np.ndarray:
        """Get the approximate covariance matrix (inverse information)."""
        pass
    
    def update(self, grad: np.ndarray, hess: Optional[np.ndarray] = None, 
               learning_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update parameter estimate with new gradient information.
        
        Args:
            grad: Gradient (score) vector
            hess: Hessian matrix (observed Fisher information)
            learning_rate: Step size scaling factor
            
        Returns:
            Updated parameter estimate and uncertainty
        """
        # Update information
        self._update_information(grad, hess)
        
        # Natural gradient step
        step = self._compute_natural_gradient(grad)
        self.theta -= learning_rate * step
        
        self.n_samples += 1
        
        return self.theta.copy(), self._get_covariance()
    
    @abstractmethod
    def _compute_natural_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Compute the natural gradient direction."""
        pass
    
    def confidence_interval(self, alpha: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence intervals for parameters.
        
        Args:
            alpha: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Lower and upper confidence bounds
        """
        from scipy import stats
        z = stats.norm.ppf((1 + alpha) / 2)
        std = np.sqrt(np.diag(self._get_covariance()))
        return self.theta - z * std, self.theta + z * std
    
    def predict(self, X: np.ndarray, link_function=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            X: Input features (n_samples x dim)
            link_function: Optional link function (e.g., sigmoid for logistic)
            
        Returns:
            Predictions and their standard deviations
        """
        mean = X @ self.theta
        
        # Predictive variance using error propagation
        cov = self._get_covariance()
        var = np.sum((X @ cov) * X, axis=1)
        std = np.sqrt(var)
        
        if link_function is not None:
            mean = link_function(mean)
            # Delta method for variance transformation
            grad = link_function(mean, derivative=True)
            std = std * grad
        
        return mean, std


class ScalarFF(FisherFlow):
    """Scalar Fisher Flow - single learning rate for all parameters (SGD-like)."""
    
    def _init_information(self):
        self.info = self.regularization
    
    def _update_information(self, grad: np.ndarray, hess: Optional[np.ndarray] = None):
        # Use squared gradient norm as proxy for information
        self.info += np.sum(grad**2)
    
    def _get_covariance(self) -> np.ndarray:
        return np.eye(self.dim) / self.info
    
    def _compute_natural_gradient(self, grad: np.ndarray) -> np.ndarray:
        return grad / self.info


class DiagonalFF(FisherFlow):
    """Diagonal Fisher Flow - per-parameter learning rates (Adam-like)."""
    
    def _init_information(self):
        self.info = np.ones(self.dim) * self.regularization
    
    def _update_information(self, grad: np.ndarray, hess: Optional[np.ndarray] = None):
        # Accumulate squared gradients (empirical Fisher diagonal)
        self.info += grad**2
    
    def _get_covariance(self) -> np.ndarray:
        return np.diag(1.0 / self.info)
    
    def _compute_natural_gradient(self, grad: np.ndarray) -> np.ndarray:
        return grad / self.info


class FullFF(FisherFlow):
    """Full Fisher Flow - complete information matrix (Natural Gradient)."""
    
    def _init_information(self):
        self.info = np.eye(self.dim) * self.regularization
    
    def _update_information(self, grad: np.ndarray, hess: Optional[np.ndarray] = None):
        if hess is not None:
            # Use provided Hessian (observed Fisher)
            self.info -= hess  # Note: Hessian is negative of Fisher
        else:
            # Use outer product approximation (empirical Fisher)
            self.info += np.outer(grad, grad)
    
    def _get_covariance(self) -> np.ndarray:
        return np.linalg.inv(self.info + np.eye(self.dim) * self.regularization)
    
    def _compute_natural_gradient(self, grad: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.info, grad)


class BlockFF(FisherFlow):
    """Block-diagonal Fisher Flow - groups of parameters share information."""
    
    def __init__(self, dim: int, block_sizes: list, regularization: float = 1e-6):
        """
        Initialize Block Fisher Flow.
        
        Args:
            dim: Total parameter dimension
            block_sizes: List of block sizes (must sum to dim)
            regularization: Numerical stability constant
        """
        assert sum(block_sizes) == dim, "Block sizes must sum to dimension"
        self.block_sizes = block_sizes
        self.block_starts = np.cumsum([0] + block_sizes[:-1])
        super().__init__(dim, regularization)
    
    def _init_information(self):
        self.info_blocks = [
            np.eye(size) * self.regularization 
            for size in self.block_sizes
        ]
    
    def _update_information(self, grad: np.ndarray, hess: Optional[np.ndarray] = None):
        for i, (start, size) in enumerate(zip(self.block_starts, self.block_sizes)):
            end = start + size
            block_grad = grad[start:end]
            
            if hess is not None:
                block_hess = hess[start:end, start:end]
                self.info_blocks[i] -= block_hess
            else:
                self.info_blocks[i] += np.outer(block_grad, block_grad)
    
    def _get_covariance(self) -> np.ndarray:
        cov = np.zeros((self.dim, self.dim))
        for i, (start, size) in enumerate(zip(self.block_starts, self.block_sizes)):
            end = start + size
            cov[start:end, start:end] = np.linalg.inv(
                self.info_blocks[i] + np.eye(size) * self.regularization
            )
        return cov
    
    def _compute_natural_gradient(self, grad: np.ndarray) -> np.ndarray:
        result = np.zeros_like(grad)
        for i, (start, size) in enumerate(zip(self.block_starts, self.block_sizes)):
            end = start + size
            result[start:end] = np.linalg.solve(
                self.info_blocks[i], grad[start:end]
            )
        return result


class KroneckerFF(FisherFlow):
    """
    Kronecker-factored Fisher Flow for matrix parameters.
    Ideal for neural network weight matrices.
    """
    
    def __init__(self, shape: Tuple[int, int], regularization: float = 1e-6):
        """
        Initialize Kronecker Fisher Flow.
        
        Args:
            shape: Shape of weight matrix (output_dim, input_dim)
            regularization: Numerical stability constant
        """
        self.shape = shape
        self.m, self.n = shape
        super().__init__(self.m * self.n, regularization)
    
    def _init_information(self):
        self.A = np.eye(self.m) * self.regularization  # Output factor
        self.B = np.eye(self.n) * self.regularization  # Input factor
    
    def _update_information(self, grad: np.ndarray, hess: Optional[np.ndarray] = None):
        # Reshape gradient to matrix form
        G = grad.reshape(self.shape)
        
        # Update Kronecker factors (simplified version)
        self.A += G @ G.T / self.n
        self.B += G.T @ G / self.m
    
    def _get_covariance(self) -> np.ndarray:
        # Full covariance would be A^{-1} ⊗ B^{-1}
        # Return as diagonal approximation for memory efficiency
        A_inv = np.linalg.inv(self.A + np.eye(self.m) * self.regularization)
        B_inv = np.linalg.inv(self.B + np.eye(self.n) * self.regularization)
        
        # Diagonal of Kronecker product
        diag = np.kron(np.diag(A_inv), np.diag(B_inv))
        return np.diag(diag)
    
    def _compute_natural_gradient(self, grad: np.ndarray) -> np.ndarray:
        G = grad.reshape(self.shape)
        
        # Natural gradient: A^{-1} G B^{-1}
        A_inv = np.linalg.inv(self.A + np.eye(self.m) * self.regularization)
        B_inv = np.linalg.inv(self.B + np.eye(self.n) * self.regularization)
        
        nat_grad = A_inv @ G @ B_inv
        return nat_grad.ravel()


class ExponentialFF(DiagonalFF):
    """Diagonal Fisher Flow with exponential forgetting for non-stationary data."""
    
    def __init__(self, dim: int, decay_rate: float = 0.99, regularization: float = 1e-6):
        """
        Initialize Exponential Fisher Flow.
        
        Args:
            dim: Parameter dimension
            decay_rate: Forgetting factor (0 < decay_rate < 1)
            regularization: Numerical stability constant
        """
        self.decay_rate = decay_rate
        super().__init__(dim, regularization)
    
    def _update_information(self, grad: np.ndarray, hess: Optional[np.ndarray] = None):
        # Decay old information
        self.info *= self.decay_rate
        # Add new information
        self.info += grad**2


# Example usage and testing
if __name__ == "__main__":
    # Simple demonstration with synthetic data
    np.random.seed(42)
    
    # Generate synthetic linear regression data
    n_samples, n_features = 1000, 10
    X = np.random.randn(n_samples, n_features)
    true_theta = np.random.randn(n_features)
    y = X @ true_theta + 0.1 * np.random.randn(n_samples)
    
    # Compare different Fisher Flow variants
    methods = {
        'Scalar': ScalarFF(n_features),
        'Diagonal': DiagonalFF(n_features),
        'Full': FullFF(n_features),
        'Exponential': ExponentialFF(n_features, decay_rate=0.95)
    }
    
    print("Fisher Flow Demonstration")
    print("=" * 50)
    
    # Online learning
    batch_size = 10
    for method_name, ff in methods.items():
        mse = 0
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # Compute gradient (for linear regression)
            pred = batch_X @ ff.theta
            error = pred - batch_y
            grad = batch_X.T @ error / batch_size
            
            # Update Fisher Flow
            theta, cov = ff.update(grad, learning_rate=0.01)
            
            # Track error
            mse += np.mean(error**2)
        
        # Final evaluation
        final_pred = X @ ff.theta
        final_mse = np.mean((final_pred - y)**2)
        param_error = np.linalg.norm(ff.theta - true_theta)
        
        print(f"\n{method_name} Fisher Flow:")
        print(f"  Final MSE: {final_mse:.4f}")
        print(f"  Parameter Error: {param_error:.4f}")
        
        # Show confidence intervals for first 3 parameters
        ci_lower, ci_upper = ff.confidence_interval(0.95)
        print(f"  95% CI for θ[0]: [{ci_lower[0]:.3f}, {ci_upper[0]:.3f}]")
        print(f"  True θ[0]: {true_theta[0]:.3f}")