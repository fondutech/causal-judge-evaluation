"""
CJE Testing Infrastructure

This module provides mock implementations and testing utilities for CJE components,
enabling full pipeline testing without external API dependencies or model loading.

Key components:
- MockPolicyRunner: Simulate local models without loading transformers
- MockAPIPolicyRunner: Simulate API calls without external dependencies
- MockJudge: Simulate judge scoring without API calls
- Test fixtures: Common test scenarios and data
- Integration helpers: Easy mock setup for full pipeline testing
"""

from .mocks.policy_runners import MockPolicyRunner, MockAPIPolicyRunner
from .mocks import MockJudge, MockAPIJudge, MockLocalJudge
from .mocks.multi_target_sampler import (
    MockMultiTargetSampler,
    create_mock_multi_sampler,
)
from .fixtures.data import *
from .fixtures.configs import *
from .fixtures.scenarios import *
from .integration import (
    enable_testing_mode,
    disable_testing_mode,
    create_mock_pipeline,
    testing_mode,
    run_mock_pipeline_test,
)

__all__ = [
    # Mock implementations
    "MockPolicyRunner",
    "MockAPIPolicyRunner",
    "MockJudge",
    "MockAPIJudge",
    "MockLocalJudge",
    "MockMultiTargetSampler",
    "create_mock_multi_sampler",
    # Integration utilities
    "enable_testing_mode",
    "disable_testing_mode",
    "testing_mode",
    "create_mock_pipeline",
    "run_mock_pipeline_test",
    # Fixtures are imported via *
]
