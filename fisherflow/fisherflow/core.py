"""Core Fisher Flow implementations."""

import numpy as np
from typing import Optional, Tuple, Union, List
from abc import ABC, abstractmethod


class FisherInformation:
    """
    Represents Fisher Information Matrix and its approximations.
    
    The Fisher Information Matrix (FIM) is the fundamental quantity in Fisher Flow,
    encoding statistical distinguishability and parameter uncertainty.
    """
    
    def __init__(self, matrix: np.ndarray, structure: str = "full"):
        """
        Initialize Fisher Information.
        
        Args:
            matrix: The Fisher Information Matrix (or its approximation)
            structure: Type of structure ("full", "diagonal", "block", "kronecker")
        """
        self.matrix = np.asarray(matrix)
        self.structure = structure
        self._validate_matrix()
        
    def _validate_matrix(self):
        """Validate that the matrix is positive semi-definite."""
        if self.structure == "diagonal":
            if np.any(self.matrix < 0):
                raise ValueError("Diagonal Fisher Information must have non-negative elements")
        elif self.structure == "full":
            eigenvalues = np.linalg.eigvalsh(self.matrix)
            if np.any(eigenvalues < -1e-10):
                raise ValueError("Fisher Information Matrix must be positive semi-definite")
    
    def add(self, other: 'FisherInformation') -> 'FisherInformation':
        """
        Add two Fisher Information matrices (information accumulation).
        
        This implements the fundamental property: I_total = I_1 + I_2
        """
        if self.structure != other.structure:
            raise ValueError(f"Cannot add Fisher Information with different structures: {self.structure} vs {other.structure}")
        
        return FisherInformation(self.matrix + other.matrix, self.structure)
    
    def inverse(self, damping: float = 1e-6) -> np.ndarray:
        """
        Compute the inverse of Fisher Information (covariance).
        
        Args:
            damping: Small value added to diagonal for numerical stability
        """
        if self.structure == "diagonal":
            return 1.0 / (self.matrix + damping)
        elif self.structure == "full":
            return np.linalg.inv(self.matrix + damping * np.eye(len(self.matrix)))
        else:
            raise NotImplementedError(f"Inverse not implemented for structure: {self.structure}")
    
    def solve(self, vector: np.ndarray, damping: float = 1e-6) -> np.ndarray:
        """
        Solve I * x = vector efficiently based on structure.
        
        This is the key operation for natural gradient: theta_new = theta_old - learning_rate * I^{-1} * gradient
        """
        if self.structure == "diagonal":
            return vector / (self.matrix + damping)
        elif self.structure == "full":
            return np.linalg.solve(self.matrix + damping * np.eye(len(self.matrix)), vector)
        else:
            raise NotImplementedError(f"Solve not implemented for structure: {self.structure}")
    
    def log_determinant(self) -> float:
        """Compute log determinant (relates to entropy)."""
        if self.structure == "diagonal":
            return np.sum(np.log(self.matrix + 1e-10))
        elif self.structure == "full":
            return np.linalg.slogdet(self.matrix)[1]
        else:
            raise NotImplementedError(f"Log determinant not implemented for structure: {self.structure}")
    
    def __repr__(self):
        return f"FisherInformation(structure={self.structure}, shape={self.matrix.shape})"


class FisherFlowEstimator(ABC):
    """
    Abstract base class for Fisher Flow estimators.
    
    Fisher Flow propagates (theta_hat, I) pairs instead of full distributions.
    """
    
    def __init__(self, dim: int, structure: str = "full", damping: float = 1e-6):
        """
        Initialize Fisher Flow estimator.
        
        Args:
            dim: Dimension of parameter space
            structure: Type of Fisher Information approximation
            damping: Numerical stability parameter
        """
        self.dim = dim
        self.structure = structure
        self.damping = damping
        
        # Initialize state: (theta_hat, I)
        self.theta = np.zeros(dim)
        self.information = self._initialize_information()
        
        # Track history for analysis
        self.history = {
            'theta': [self.theta.copy()],
            'information_norm': [np.linalg.norm(self.information.matrix)]
        }
    
    def _initialize_information(self) -> FisherInformation:
        """Initialize Fisher Information based on structure."""
        if self.structure == "diagonal":
            return FisherInformation(np.ones(self.dim) * 1e-4, "diagonal")
        elif self.structure == "full":
            return FisherInformation(np.eye(self.dim) * 1e-4, "full")
        else:
            raise NotImplementedError(f"Initialization not implemented for structure: {self.structure}")
    
    @abstractmethod
    def compute_score(self, data: np.ndarray) -> np.ndarray:
        """
        Compute score function (gradient of log-likelihood).
        
        Args:
            data: Batch of data
            
        Returns:
            Score vector
        """
        pass
    
    @abstractmethod
    def compute_fisher_information(self, data: np.ndarray) -> FisherInformation:
        """
        Compute Fisher Information from data batch.
        
        Args:
            data: Batch of data
            
        Returns:
            Fisher Information for this batch
        """
        pass
    
    def update(self, data: np.ndarray, learning_rate: float = 1.0):
        """
        Perform Fisher Flow update with new data batch.
        
        This implements the core Fisher Flow equations:
        1. I_t = I_{t-1} + I_{batch}  (information accumulation)
        2. theta_t = theta_{t-1} - lr * I_t^{-1} * score  (natural gradient step)
        """
        # Compute score and Fisher Information for batch
        score = self.compute_score(data)
        batch_information = self.compute_fisher_information(data)
        
        # Information accumulation (additive property)
        self.information = self.information.add(batch_information)
        
        # Natural gradient update
        update_direction = self.information.solve(score, self.damping)
        self.theta = self.theta - learning_rate * update_direction
        
        # Record history
        self.history['theta'].append(self.theta.copy())
        self.history['information_norm'].append(np.linalg.norm(self.information.matrix))
    
    def get_uncertainty(self) -> np.ndarray:
        """
        Get parameter uncertainty (inverse Fisher Information).
        
        Returns:
            Covariance matrix or diagonal variances
        """
        return self.information.inverse(self.damping)
    
    def get_confidence_interval(self, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence intervals for parameters.
        
        Args:
            alpha: Significance level (default 0.05 for 95% CI)
            
        Returns:
            Lower and upper bounds for each parameter
        """
        from scipy import stats
        
        uncertainty = self.get_uncertainty()
        if self.structure == "diagonal":
            std_errors = np.sqrt(uncertainty)
        else:
            std_errors = np.sqrt(np.diag(uncertainty))
        
        z_score = stats.norm.ppf(1 - alpha/2)
        lower = self.theta - z_score * std_errors
        upper = self.theta + z_score * std_errors
        
        return lower, upper
    
    def reset(self):
        """Reset estimator to initial state."""
        self.theta = np.zeros(self.dim)
        self.information = self._initialize_information()
        self.history = {
            'theta': [self.theta.copy()],
            'information_norm': [np.linalg.norm(self.information.matrix)]
        }