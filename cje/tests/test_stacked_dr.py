"""Tests for StackedDREstimator - optimal stacking of DR methods."""

import numpy as np
import pytest
from unittest.mock import Mock
from typing import Dict, Any

from cje.estimators.stacking import StackedDREstimator
from cje.data.models import EstimationResult


class TestStackedDR:
    """Test suite for StackedDREstimator - focuses on key functionality only."""
    
    def test_basic_stacking_workflow(self):
        """Test the core stacking workflow with mock estimators."""
        # Create mock sampler
        sampler = Mock()
        sampler.oracle_coverage = 0.5
        sampler.n_valid_samples = 100
        sampler.target_policies = ["policy_a", "policy_b"]
        
        # Initialize stacked estimator
        stacked = StackedDREstimator(
            sampler,
            estimators=["dr-cpo", "tmle"],
            use_outer_split=False,  # Simpler for testing
            parallel=False
        )
        
        # Create mock component results with different variances
        # This tests that stacking prefers lower-variance estimators
        n = 100
        base_if = np.random.randn(n)
        
        # DR-CPO: Higher variance
        result_drcpo = Mock()
        result_drcpo.estimates = np.array([0.5, 0.6])
        result_drcpo.standard_errors = np.array([0.10, 0.12])  # Higher SE
        result_drcpo.influence_functions = {
            "policy_a": base_if + 0.5 * np.random.randn(n),
            "policy_b": base_if + 0.6 * np.random.randn(n)
        }
        
        # TMLE: Lower variance
        result_tmle = Mock()
        result_tmle.estimates = np.array([0.52, 0.61])
        result_tmle.standard_errors = np.array([0.08, 0.09])  # Lower SE
        result_tmle.influence_functions = {
            "policy_a": base_if + 0.2 * np.random.randn(n),  # Less noise
            "policy_b": base_if + 0.3 * np.random.randn(n)
        }
        
        # Mock the estimator running
        stacked.component_results = {
            "dr-cpo": result_drcpo,
            "tmle": result_tmle
        }
        
        # Test weight computation
        IF_matrix = np.column_stack([
            result_drcpo.influence_functions["policy_a"],
            result_tmle.influence_functions["policy_a"]
        ])
        weights = stacked._compute_optimal_weights(IF_matrix)
        
        # Verify properties
        assert np.allclose(weights.sum(), 1.0), "Weights must sum to 1"
        assert np.all(weights >= 0), "Weights must be non-negative"
        assert weights[1] > weights[0], "Lower-variance TMLE should get higher weight"
    
    def test_fallback_to_single_estimator(self):
        """Test graceful fallback when only one estimator succeeds."""
        sampler = Mock()
        sampler.oracle_coverage = None
        sampler.n_valid_samples = 100
        sampler.target_policies = ["policy_a"]
        
        stacked = StackedDREstimator(sampler, estimators=["dr-cpo", "tmle"])
        
        # Only one estimator succeeded
        successful_result = Mock()
        successful_result.estimates = np.array([0.5])
        successful_result.standard_errors = np.array([0.1])
        successful_result.influence_functions = {"policy_a": np.random.randn(100)}
        successful_result.diagnostics = None
        successful_result.metadata = Mock()
        successful_result.metadata.copy = Mock(return_value={})
        
        stacked.component_results = {
            "dr-cpo": successful_result,
            "tmle": None  # Failed
        }
        
        # Should identify only one valid estimator
        valid = stacked._get_valid_estimators()
        assert valid == ["dr-cpo"]
        
        # Passthrough creation should add metadata
        from unittest.mock import patch
        with patch('cje.estimators.stacking.EstimationResult') as MockResult:
            MockResult.return_value = Mock()
            passthrough = stacked._create_passthrough_result("dr-cpo")
            
            # Check that metadata indicates fallback
            call_kwargs = MockResult.call_args[1]
            assert call_kwargs["metadata"]["stacking_fallback"] is True
            assert call_kwargs["metadata"]["selected_estimator"] == "dr-cpo"
    
    def test_outer_split_for_honest_inference(self):
        """Test that outer split produces valid stacked influence functions."""
        sampler = Mock()
        sampler.oracle_coverage = None
        sampler.n_valid_samples = 90  # Divisible by folds
        sampler.target_policies = ["policy_a"]
        
        stacked = StackedDREstimator(sampler, use_outer_split=True, V_folds=3)
        
        # Create influence functions for two estimators
        n = 90
        IF_matrix = np.random.randn(n, 2)
        
        # Run outer split stacking
        stacked_if, avg_weights = stacked._stack_with_outer_split(IF_matrix, "policy_a")
        
        # Verify outputs
        assert stacked_if.shape == (n,), "Should return IF for all samples"
        assert not np.any(np.isnan(stacked_if)), "No NaN values in stacked IF"
        assert avg_weights.shape == (2,), "Should return weights for both estimators"
        assert np.allclose(avg_weights.sum(), 1.0), "Weights should sum to 1"
    
    def test_diagnostic_reporting(self):
        """Test that stacking produces appropriate diagnostics."""
        sampler = Mock()
        sampler.oracle_coverage = None
        sampler.n_valid_samples = 100
        sampler.target_policies = ["policy_a"]
        
        stacked = StackedDREstimator(sampler, estimators=["dr-cpo", "tmle", "mrdr"])
        stacked.weights_per_policy = {"policy_a": np.array([0.2, 0.5, 0.3])}
        
        # Create mock results to avoid KeyError
        for name in ["dr-cpo", "tmle", "mrdr"]:
            mock_result = Mock()
            mock_result.influence_functions = None  # No IFs to avoid correlation matrix
            stacked.component_results[name] = mock_result
        
        # Build diagnostics
        diagnostics = stacked._build_stacking_diagnostics(["dr-cpo", "tmle", "mrdr"])
        
        # Verify structure
        assert diagnostics["estimator_type"] == "StackedDR"
        assert diagnostics["n_components"] == 3
        assert diagnostics["valid_estimators"] == ["dr-cpo", "tmle", "mrdr"]
        assert "weights_per_policy" in diagnostics
        
        # Check weight reporting
        policy_weights = diagnostics["weights_per_policy"]["policy_a"]
        assert policy_weights["dr-cpo"] == 0.2
        assert policy_weights["tmle"] == 0.5
        assert policy_weights["mrdr"] == 0.3