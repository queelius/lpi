"""
Fisher Flow Optimizers - Various approximations of the Fisher Flow framework.

This module demonstrates how popular optimization algorithms are special cases
of Fisher Flow with different approximation structures.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from .autograd import Value, Module


class FisherFlowOptimizer:
    """Base class for Fisher Flow optimizers."""
    
    def __init__(self, params: List[Value], lr: float = 0.01, damping: float = 1e-6):
        """
        Initialize Fisher Flow optimizer.
        
        Args:
            params: List of parameters (Value objects)
            lr: Learning rate
            damping: Numerical stability parameter
        """
        self.params = params
        self.lr = lr
        self.damping = damping
        self.t = 0  # Time step
        
    def step(self):
        """Perform one optimization step."""
        raise NotImplementedError
    
    def zero_grad(self):
        """Zero out gradients."""
        for p in self.params:
            p.grad = 0
            p.fisher_diag = 0


class DiagonalFisherFlow(FisherFlowOptimizer):
    """
    Diagonal Fisher Flow optimizer.
    
    This approximates the Fisher Information Matrix with only diagonal elements,
    similar to Adam but with proper information-geometric interpretation.
    """
    
    def __init__(self, params: List[Value], lr: float = 0.01, damping: float = 1e-6,
                 beta: float = 0.999):
        """
        Initialize Diagonal Fisher Flow.
        
        Args:
            params: List of parameters
            lr: Learning rate
            damping: Numerical stability
            beta: Exponential decay rate for Fisher accumulation
        """
        super().__init__(params, lr, damping)
        self.beta = beta
        
        # Initialize diagonal Fisher Information for each parameter
        self.fisher_diag = {p: damping for p in params}
        
    def step(self):
        """
        Perform diagonal Fisher Flow update.
        
        Update rule: θ_t+1 = θ_t - η * F_diag^{-1} * ∇L
        where F_diag is the diagonal Fisher Information.
        """
        self.t += 1
        
        for p in self.params:
            if p.grad == 0:
                continue
                
            # Accumulate Fisher Information (exponential moving average)
            self.fisher_diag[p] = (self.beta * self.fisher_diag[p] + 
                                  (1 - self.beta) * p.grad ** 2)
            
            # Bias correction (similar to Adam)
            fisher_corrected = self.fisher_diag[p] / (1 - self.beta ** self.t)
            
            # Natural gradient step with diagonal Fisher
            p.data -= self.lr * p.grad / (np.sqrt(fisher_corrected) + self.damping)


class AdamAsFisherFlow(FisherFlowOptimizer):
    """
    Adam optimizer reinterpreted as Diagonal Fisher Flow.
    
    This shows that Adam is essentially diagonal Fisher Flow with:
    - First moment (momentum) as exponential moving average of gradients
    - Second moment as diagonal Fisher Information estimate
    """
    
    def __init__(self, params: List[Value], lr: float = 0.001, 
                 beta1: float = 0.9, beta2: float = 0.999, damping: float = 1e-8):
        """
        Initialize Adam as Fisher Flow.
        
        Args:
            params: List of parameters
            lr: Learning rate
            beta1: Decay rate for first moment (momentum)
            beta2: Decay rate for second moment (Fisher diagonal)
            damping: Numerical stability (epsilon in Adam)
        """
        super().__init__(params, lr, damping)
        self.beta1 = beta1
        self.beta2 = beta2
        
        # Initialize moments
        self.m = {p: 0 for p in params}  # First moment (momentum)
        self.v = {p: 0 for p in params}  # Second moment (diagonal Fisher)
        
    def step(self):
        """
        Perform Adam update (diagonal Fisher Flow with momentum).
        
        This is mathematically equivalent to:
        - Estimating diagonal Fisher Information via second moments
        - Using momentum for the gradient estimate
        - Applying natural gradient with diagonal Fisher
        """
        self.t += 1
        
        for p in self.params:
            if p.grad == 0:
                continue
            
            # Update biased first moment (momentum)
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * p.grad
            
            # Update biased second moment (diagonal Fisher estimate)
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (p.grad ** 2)
            
            # Bias correction
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            
            # Natural gradient step with diagonal Fisher
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.damping)


class NaturalGradientFlow(FisherFlowOptimizer):
    """
    Natural Gradient optimizer - Exact Fisher Flow with full Fisher Information Matrix.
    
    This is the theoretically optimal but computationally expensive version.
    """
    
    def __init__(self, params: List[Value], lr: float = 0.01, damping: float = 1e-4,
                 update_freq: int = 10):
        """
        Initialize Natural Gradient (full Fisher Flow).
        
        Args:
            params: List of parameters
            lr: Learning rate
            damping: Tikhonov damping for Fisher matrix inversion
            update_freq: How often to recompute Fisher matrix
        """
        super().__init__(params, lr, damping)
        self.update_freq = update_freq
        self.n_params = len(params)
        
        # Initialize full Fisher Information Matrix
        self.fisher_matrix = np.eye(self.n_params) * damping
        
    def compute_fisher_matrix(self, model: Module, data_loader, loss_fn):
        """
        Compute empirical Fisher Information Matrix.
        
        Args:
            model: The model containing parameters
            data_loader: Iterator over data batches
            loss_fn: Loss function
        """
        fisher_sum = np.zeros((self.n_params, self.n_params))
        n_samples = 0
        
        for batch in data_loader:
            model.zero_grad()
            loss = loss_fn(batch, model)
            loss.backward()
            
            # Collect gradients
            grads = np.array([p.grad for p in self.params])
            
            # Empirical Fisher: F = E[grad * grad^T]
            fisher_sum += np.outer(grads, grads)
            n_samples += 1
        
        self.fisher_matrix = fisher_sum / n_samples + self.damping * np.eye(self.n_params)
    
    def step(self):
        """
        Perform natural gradient step.
        
        Update rule: θ_t+1 = θ_t - η * F^{-1} * ∇L
        where F is the full Fisher Information Matrix.
        """
        self.t += 1
        
        # Collect gradients
        grads = np.array([p.grad for p in self.params])
        
        # Natural gradient direction: F^{-1} * grad
        try:
            natural_grad = np.linalg.solve(self.fisher_matrix, grads)
        except np.linalg.LinAlgError:
            # Fall back to gradient descent if Fisher is singular
            natural_grad = grads
        
        # Update parameters
        for i, p in enumerate(self.params):
            p.data -= self.lr * natural_grad[i]


class KroneckerFisherFlow(FisherFlowOptimizer):
    """
    Kronecker-Factored Approximate Curvature (K-FAC) as Fisher Flow.
    
    This approximates the Fisher Information Matrix for neural network layers
    using Kronecker products, dramatically reducing computational cost.
    """
    
    def __init__(self, params: List[Value], lr: float = 0.01, damping: float = 1e-4,
                 momentum: float = 0.9, update_freq: int = 10):
        """
        Initialize K-FAC optimizer.
        
        Args:
            params: List of parameters  
            lr: Learning rate
            damping: Tikhonov damping
            momentum: Momentum coefficient
            update_freq: How often to update Fisher factors
        """
        super().__init__(params, lr, damping)
        self.momentum = momentum
        self.update_freq = update_freq
        
        # For simplified implementation, we use block-diagonal approximation
        # In full K-FAC, this would use Kronecker factors for each layer
        self.momentum_buffer = {p: 0 for p in params}
        self.fisher_moving_avg = {p: self.damping for p in params}
        
    def step(self):
        """
        Perform K-FAC update (Kronecker-factored Fisher Flow).
        
        In a full implementation, this would:
        1. Maintain Kronecker factors A and B for each layer
        2. Update: (A ⊗ B)^{-1} = A^{-1} ⊗ B^{-1}
        3. Apply natural gradient with Kronecker structure
        """
        self.t += 1
        
        for p in self.params:
            if p.grad == 0:
                continue
            
            # Simplified K-FAC: use diagonal approximation with momentum
            # Full K-FAC would use Kronecker factors here
            
            # Update Fisher moving average
            if self.t % self.update_freq == 0:
                self.fisher_moving_avg[p] = (0.95 * self.fisher_moving_avg[p] + 
                                            0.05 * (p.grad ** 2))
            
            # Compute preconditioned gradient
            preconditioned_grad = p.grad / (np.sqrt(self.fisher_moving_avg[p]) + self.damping)
            
            # Apply momentum
            self.momentum_buffer[p] = (self.momentum * self.momentum_buffer[p] + 
                                      (1 - self.momentum) * preconditioned_grad)
            
            # Update parameter
            p.data -= self.lr * self.momentum_buffer[p]


class ElasticWeightConsolidation(FisherFlowOptimizer):
    """
    Elastic Weight Consolidation (EWC) as Fisher Flow regularization.
    
    EWC uses Fisher Information to prevent catastrophic forgetting in continual learning
    by penalizing changes to important parameters.
    """
    
    def __init__(self, params: List[Value], lr: float = 0.01, 
                 ewc_lambda: float = 1000, damping: float = 1e-6):
        """
        Initialize EWC optimizer.
        
        Args:
            params: List of parameters
            lr: Learning rate
            ewc_lambda: Regularization strength
            damping: Numerical stability
        """
        super().__init__(params, lr, damping)
        self.ewc_lambda = ewc_lambda
        
        # Store Fisher Information and optimal parameters from previous task
        self.fisher_diag_prev = {p: 0 for p in params}
        self.params_prev = {p: p.data for p in params}
        
    def consolidate(self, model: Module, data_loader, loss_fn):
        """
        Consolidate current task knowledge by computing Fisher Information.
        
        Args:
            model: Current model
            data_loader: Data from current task
            loss_fn: Loss function for current task
        """
        # Compute diagonal Fisher Information for current task
        fisher_accumulator = {p: 0 for p in self.params}
        n_samples = 0
        
        for batch in data_loader:
            model.zero_grad()
            loss = loss_fn(batch, model)
            loss.backward()
            
            for p in self.params:
                fisher_accumulator[p] += p.grad ** 2
            n_samples += 1
        
        # Store Fisher Information and current parameters
        for p in self.params:
            self.fisher_diag_prev[p] = fisher_accumulator[p] / n_samples
            self.params_prev[p] = p.data
    
    def step(self):
        """
        Perform EWC update with Fisher Information regularization.
        
        The loss becomes: L_new + λ/2 * Σ F_i * (θ_i - θ*_i)^2
        where F_i is Fisher Information and θ*_i are parameters from previous task.
        """
        self.t += 1
        
        for p in self.params:
            if p.grad == 0:
                continue
            
            # Add EWC penalty gradient
            ewc_grad = self.ewc_lambda * self.fisher_diag_prev[p] * (p.data - self.params_prev[p])
            
            # Combined gradient
            total_grad = p.grad + ewc_grad
            
            # Standard gradient descent step
            p.data -= self.lr * total_grad