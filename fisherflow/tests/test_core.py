"""Tests for core Fisher Flow functionality."""

import numpy as np
import pytest
import sys
sys.path.append('..')

from fisherflow.core import FisherInformation, FisherFlowEstimator


class TestFisherInformation:
    """Test Fisher Information Matrix operations."""
    
    def test_initialization(self):
        """Test Fisher Information initialization."""
        # Diagonal structure
        diag = np.array([1.0, 2.0, 3.0])
        fi_diag = FisherInformation(diag, structure="diagonal")
        assert fi_diag.structure == "diagonal"
        assert np.allclose(fi_diag.matrix, diag)
        
        # Full structure
        full = np.array([[2.0, 0.5], [0.5, 1.0]])
        fi_full = FisherInformation(full, structure="full")
        assert fi_full.structure == "full"
        assert np.allclose(fi_full.matrix, full)
    
    def test_positive_definiteness_check(self):
        """Test that non-PSD matrices raise errors."""
        # Negative diagonal element
        with pytest.raises(ValueError, match="non-negative"):
            FisherInformation(np.array([-1.0, 2.0]), structure="diagonal")
        
        # Non-PSD full matrix
        non_psd = np.array([[1.0, 2.0], [2.0, 1.0]])  # Eigenvalues: -1, 3
        with pytest.raises(ValueError, match="positive semi-definite"):
            FisherInformation(non_psd, structure="full")
    
    def test_information_addition(self):
        """Test information accumulation property."""
        # Diagonal case
        fi1 = FisherInformation(np.array([1.0, 2.0]), "diagonal")
        fi2 = FisherInformation(np.array([3.0, 4.0]), "diagonal")
        fi_sum = fi1.add(fi2)
        assert np.allclose(fi_sum.matrix, [4.0, 6.0])
        
        # Full case
        fi1 = FisherInformation(np.eye(2), "full")
        fi2 = FisherInformation(np.eye(2) * 2, "full")
        fi_sum = fi1.add(fi2)
        assert np.allclose(fi_sum.matrix, np.eye(2) * 3)
    
    def test_inverse(self):
        """Test Fisher Information inverse (covariance)."""
        # Diagonal case
        fi = FisherInformation(np.array([4.0, 9.0]), "diagonal")
        inv = fi.inverse(damping=0)
        assert np.allclose(inv, [0.25, 1/9])
        
        # Full case with damping
        fi = FisherInformation(np.eye(2) * 2, "full")
        inv = fi.inverse(damping=0.1)
        expected = np.linalg.inv(np.eye(2) * 2.1)
        assert np.allclose(inv, expected)
    
    def test_solve(self):
        """Test solving linear system with Fisher Information."""
        # Diagonal case
        fi = FisherInformation(np.array([2.0, 4.0]), "diagonal")
        vector = np.array([1.0, 2.0])
        solution = fi.solve(vector, damping=0)
        assert np.allclose(solution, [0.5, 0.5])
        
        # Full case
        fi = FisherInformation(np.array([[2.0, 0], [0, 4.0]]), "full")
        solution = fi.solve(vector, damping=0)
        assert np.allclose(solution, [0.5, 0.5])
    
    def test_log_determinant(self):
        """Test log determinant computation."""
        # Diagonal case
        fi = FisherInformation(np.array([2.0, 3.0]), "diagonal")
        log_det = fi.log_determinant()
        expected = np.log(2.0) + np.log(3.0)
        assert np.isclose(log_det, expected)
        
        # Full case
        matrix = np.array([[2.0, 0.5], [0.5, 3.0]])
        fi = FisherInformation(matrix, "full")
        log_det = fi.log_determinant()
        expected = np.linalg.slogdet(matrix)[1]
        assert np.isclose(log_det, expected)


class SimpleEstimator(FisherFlowEstimator):
    """Simple estimator for testing."""
    
    def compute_score(self, data):
        """Simple linear score."""
        return np.ones(self.dim) * data.mean()
    
    def compute_fisher_information(self, data):
        """Simple Fisher Information."""
        if self.structure == "diagonal":
            return FisherInformation(np.ones(self.dim) * data.var(), "diagonal")
        else:
            return FisherInformation(np.eye(self.dim) * data.var(), "full")


class TestFisherFlowEstimator:
    """Test Fisher Flow Estimator base functionality."""
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = SimpleEstimator(dim=3, structure="diagonal")
        assert estimator.dim == 3
        assert estimator.structure == "diagonal"
        assert len(estimator.theta) == 3
        assert np.allclose(estimator.theta, 0)
    
    def test_update(self):
        """Test Fisher Flow update step."""
        estimator = SimpleEstimator(dim=2, structure="diagonal")
        data = np.array([1.0, 2.0, 3.0])
        
        # Initial state
        theta_init = estimator.theta.copy()
        
        # Update
        estimator.update(data, learning_rate=0.1)
        
        # Check that parameters changed
        assert not np.allclose(estimator.theta, theta_init)
        
        # Check history is recorded
        assert len(estimator.history['theta']) == 2
        assert len(estimator.history['information_norm']) == 2
    
    def test_uncertainty_quantification(self):
        """Test uncertainty estimation."""
        estimator = SimpleEstimator(dim=2, structure="diagonal")
        
        # Update with some data
        for _ in range(5):
            data = np.random.randn(10)
            estimator.update(data, learning_rate=0.1)
        
        # Get uncertainty
        uncertainty = estimator.get_uncertainty()
        assert uncertainty.shape == (2,)  # Diagonal case
        assert np.all(uncertainty > 0)  # Positive variances
    
    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        estimator = SimpleEstimator(dim=3, structure="diagonal")
        
        # Update with data
        for _ in range(10):
            data = np.random.randn(20)
            estimator.update(data, learning_rate=0.05)
        
        # Get confidence intervals
        lower, upper = estimator.get_confidence_interval(alpha=0.05)
        
        assert len(lower) == 3
        assert len(upper) == 3
        assert np.all(upper > lower)  # Upper bounds > lower bounds
        
        # Check that current estimate is within CI
        assert np.all(estimator.theta >= lower)
        assert np.all(estimator.theta <= upper)
    
    def test_reset(self):
        """Test estimator reset."""
        estimator = SimpleEstimator(dim=2)
        
        # Update state
        estimator.update(np.array([1, 2, 3]), learning_rate=0.1)
        assert not np.allclose(estimator.theta, 0)
        
        # Reset
        estimator.reset()
        assert np.allclose(estimator.theta, 0)
        assert len(estimator.history['theta']) == 1
    
    def test_information_accumulation(self):
        """Test that information accumulates correctly."""
        estimator = SimpleEstimator(dim=2, structure="diagonal")
        
        # Track information norm
        info_norms = [estimator.history['information_norm'][0]]
        
        # Multiple updates
        for _ in range(5):
            data = np.random.randn(10) + 1.0  # Positive variance
            estimator.update(data, learning_rate=0.01)
            info_norms.append(estimator.history['information_norm'][-1])
        
        # Information should be non-decreasing (monotonicity)
        for i in range(1, len(info_norms)):
            assert info_norms[i] >= info_norms[i-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])