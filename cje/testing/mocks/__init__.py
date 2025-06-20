"""Mock implementations for CJE components - unified version."""

from .policy_runners import MockPolicyRunner, MockAPIPolicyRunner
from .multi_target_sampler import MockMultiTargetSampler

# Import unified mock judges
from .judges import (
    MockJudge,
    DeterministicMockJudge,
    MCMockJudge,
    LenientJudge,
    HarshJudge,
    NoisyJudge,
    RandomJudge,
    ConstantJudge,
    MockJudgeConfig,
    create_mock_judge,
)

# For backward compatibility
MockAPIJudge = MockJudge  # They're the same now
MockLocalJudge = MockJudge

__all__ = [
    "MockPolicyRunner",
    "MockAPIPolicyRunner",
    # Unified mock judges
    "MockJudge",
    "DeterministicMockJudge",
    "MCMockJudge",
    # Personality judges
    "LenientJudge",
    "HarshJudge",
    "NoisyJudge",
    "RandomJudge",
    "ConstantJudge",
    # Config and factory
    "MockJudgeConfig",
    "create_mock_judge",
    # Backward compatibility
    "MockAPIJudge",
    "MockLocalJudge",
    # Other mocks
    "MockMultiTargetSampler",
]
