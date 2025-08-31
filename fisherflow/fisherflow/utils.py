"""Utility functions for Fisher Flow computations."""

import numpy as np
from typing import List, Callable, Optional, Tuple
from .autograd import Value, Module


def compute_empirical_fisher(model: Module, data_loader, loss_fn, 
                            structure: str = "diagonal") -> np.ndarray:
    """
    Compute empirical Fisher Information Matrix.
    
    The empirical Fisher is computed as:
    F = 1/N * Σ_i grad_i * grad_i^T
    
    Args:
        model: Neural network model
        data_loader: Iterator over data batches
        loss_fn: Loss function (takes batch and model, returns Value)
        structure: "diagonal" or "full"
        
    Returns:
        Fisher Information Matrix (diagonal or full)
    """
    params = model.parameters()
    n_params = len(params)
    n_samples = 0
    
    if structure == "diagonal":
        fisher = np.zeros(n_params)
        
        for batch in data_loader:
            model.zero_grad()
            loss = loss_fn(batch, model)
            loss.backward()
            
            # Accumulate squared gradients
            for i, p in enumerate(params):
                fisher[i] += p.grad ** 2
            n_samples += 1
            
        return fisher / n_samples
        
    elif structure == "full":
        fisher = np.zeros((n_params, n_params))
        
        for batch in data_loader:
            model.zero_grad()
            loss = loss_fn(batch, model)
            loss.backward()
            
            # Collect gradients
            grads = np.array([p.grad for p in params])
            
            # Outer product
            fisher += np.outer(grads, grads)
            n_samples += 1
            
        return fisher / n_samples
    
    else:
        raise ValueError(f"Unknown structure: {structure}")


def compute_expected_fisher(model: Module, data_loader, log_prob_fn,
                           n_samples: int = 100) -> np.ndarray:
    """
    Compute expected Fisher Information Matrix via sampling.
    
    The expected Fisher is:
    F = E_x[E_{y~p(y|x,θ)}[(∇log p(y|x,θ))(∇log p(y|x,θ))^T]]
    
    Args:
        model: Neural network model
        data_loader: Iterator over input data
        log_prob_fn: Function that computes log p(y|x,θ)
        n_samples: Number of samples for Monte Carlo estimation
        
    Returns:
        Expected Fisher Information Matrix
    """
    params = model.parameters()
    n_params = len(params)
    fisher = np.zeros((n_params, n_params))
    n_data = 0
    
    for x_batch in data_loader:
        for _ in range(n_samples):
            model.zero_grad()
            
            # Sample y from model's predictive distribution
            y_sample = sample_from_model(model, x_batch)
            
            # Compute log probability
            log_prob = log_prob_fn(x_batch, y_sample, model)
            log_prob.backward()
            
            # Collect gradients
            grads = np.array([p.grad for p in params])
            
            # Accumulate outer product
            fisher += np.outer(grads, grads)
        
        n_data += len(x_batch)
    
    return fisher / (n_data * n_samples)


def sample_from_model(model: Module, x_batch) -> np.ndarray:
    """
    Sample from model's predictive distribution.
    
    This is a placeholder - actual implementation depends on model type.
    """
    # For classification: sample from categorical
    # For regression: sample from Gaussian
    # This is problem-specific
    output = model(x_batch)
    if isinstance(output, list):
        # Multi-output
        probs = softmax([o.data for o in output])
        return np.random.choice(len(probs), p=probs)
    else:
        # Single output (regression)
        return output.data + np.random.randn() * 0.1


def softmax(x: List[float]) -> np.ndarray:
    """Compute softmax probabilities."""
    x = np.array(x)
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def fisher_vector_product(model: Module, vector: np.ndarray, 
                         data_loader, loss_fn, damping: float = 1e-4) -> np.ndarray:
    """
    Compute Fisher-vector product F*v without forming F explicitly.
    
    Uses the identity: F*v = E[grad * (grad^T * v)]
    
    Args:
        model: Neural network model
        vector: Vector to multiply with Fisher
        data_loader: Data iterator
        loss_fn: Loss function
        damping: Tikhonov damping
        
    Returns:
        Fisher-vector product
    """
    params = model.parameters()
    n_samples = 0
    fvp = np.zeros_like(vector)
    
    for batch in data_loader:
        model.zero_grad()
        loss = loss_fn(batch, model)
        loss.backward()
        
        # Collect gradients
        grads = np.array([p.grad for p in params])
        
        # Compute grad^T * v (scalar)
        grad_dot_v = np.dot(grads, vector)
        
        # Accumulate grad * (grad^T * v)
        fvp += grads * grad_dot_v
        n_samples += 1
    
    # Average and add damping
    return fvp / n_samples + damping * vector


def conjugate_gradient(fisher_vector_product_fn: Callable, b: np.ndarray,
                       max_iter: int = 10, tol: float = 1e-6) -> np.ndarray:
    """
    Solve F*x = b using Conjugate Gradient method.
    
    This allows solving with the Fisher matrix without forming it explicitly.
    
    Args:
        fisher_vector_product_fn: Function that computes F*v
        b: Right-hand side vector
        max_iter: Maximum CG iterations
        tol: Convergence tolerance
        
    Returns:
        Solution x such that F*x ≈ b
    """
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    r_dot_r = np.dot(r, r)
    
    for _ in range(max_iter):
        if r_dot_r < tol:
            break
            
        Ap = fisher_vector_product_fn(p)
        alpha = r_dot_r / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        
        r_dot_r_new = np.dot(r, r)
        beta = r_dot_r_new / r_dot_r
        p = r + beta * p
        r_dot_r = r_dot_r_new
    
    return x


def information_gain(fisher_old: np.ndarray, fisher_new: np.ndarray) -> float:
    """
    Compute information gain between two Fisher matrices.
    
    Uses the formula: IG = 0.5 * (tr(F_new * F_old^{-1}) - log|F_new * F_old^{-1}| - d)
    
    Args:
        fisher_old: Previous Fisher Information Matrix
        fisher_new: New Fisher Information Matrix
        
    Returns:
        Information gain (KL divergence between induced Gaussians)
    """
    d = len(fisher_old)
    
    # Add small damping for numerical stability
    fisher_old_inv = np.linalg.inv(fisher_old + 1e-6 * np.eye(d))
    product = fisher_new @ fisher_old_inv
    
    trace = np.trace(product)
    sign, logdet = np.linalg.slogdet(product)
    
    if sign <= 0:
        # Matrix is not positive definite
        return float('inf')
    
    return 0.5 * (trace - logdet - d)


def fisher_rao_distance(theta1: np.ndarray, fisher1: np.ndarray,
                        theta2: np.ndarray, fisher2: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between two parameter distributions.
    
    This is the geodesic distance on the statistical manifold.
    
    Args:
        theta1: First parameter vector
        fisher1: Fisher Information at theta1
        theta2: Second parameter vector
        fisher2: Fisher Information at theta2
        
    Returns:
        Fisher-Rao distance
    """
    # Approximate using average Fisher Information
    fisher_avg = 0.5 * (fisher1 + fisher2)
    
    # Add damping for stability
    fisher_avg += 1e-6 * np.eye(len(fisher_avg))
    
    # Compute Mahalanobis distance with average Fisher metric
    diff = theta2 - theta1
    distance_squared = diff @ fisher_avg @ diff
    
    return np.sqrt(max(0, distance_squared))


def estimate_condition_number(fisher: np.ndarray) -> float:
    """
    Estimate condition number of Fisher Information Matrix.
    
    High condition number indicates ill-conditioning.
    
    Args:
        fisher: Fisher Information Matrix
        
    Returns:
        Condition number (ratio of largest to smallest eigenvalue)
    """
    eigenvalues = np.linalg.eigvalsh(fisher)
    
    # Filter out very small eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    if len(eigenvalues) == 0:
        return float('inf')
    
    return eigenvalues[-1] / eigenvalues[0]


def project_to_positive_definite(matrix: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
    """
    Project a matrix to the nearest positive definite matrix.
    
    Args:
        matrix: Input matrix (possibly indefinite)
        min_eigenvalue: Minimum eigenvalue threshold
        
    Returns:
        Positive definite matrix
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Threshold eigenvalues
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    
    # Reconstruct
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T