"""
Pytest configuration for CJE comprehensive test suite.

This module configures pytest for the comprehensive testing infrastructure
including custom markers, test organization, and performance tracking.
"""

import pytest
import numpy as np
import time
from typing import Dict, Any, Generator, List
import warnings


def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers for CJE test suite."""
    config.addinivalue_line(
        "markers",
        "theoretical: Tests for theoretical guarantees (unbiasedness, efficiency, etc.)",
    )
    config.addinivalue_line(
        "markers", "empirical: Tests for empirical claims (Arena-Hard, speedup, etc.)"
    )
    config.addinivalue_line(
        "markers", "property: Property-based tests using hypothesis"
    )
    config.addinivalue_line("markers", "integration: End-to-end integration tests")
    config.addinivalue_line("markers", "slow: Tests that take significant time to run")
    config.addinivalue_line(
        "markers", "paper_validation: Tests that directly validate paper claims"
    )
    config.addinivalue_line(
        "markers", "robustness: Tests for edge cases and robustness"
    )


@pytest.fixture(scope="session")
def test_session_info() -> Dict[str, Any]:
    """Provide session-wide test information."""
    return {
        "start_time": time.time(),
        "test_counts": {
            "theoretical": 0,
            "empirical": 0,
            "property": 0,
            "integration": 0,
            "passed": 0,
            "failed": 0,
        },
    }


@pytest.fixture
def suppress_warnings() -> Generator[None, None, None]:
    """Suppress known warnings during testing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        yield


@pytest.fixture
def random_seed() -> int:
    """Provide reproducible random seed for tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def small_test_config() -> Dict[str, Any]:
    """Configuration for fast testing during development."""
    return {
        "n_samples": 50,
        "n_trials": 5,
        "k_folds": 3,
        "timeout": 30,  # seconds
    }


@pytest.fixture
def full_test_config() -> Dict[str, Any]:
    """Configuration for comprehensive testing."""
    return {
        "n_samples": 500,
        "n_trials": 50,
        "k_folds": 5,
        "timeout": 300,  # seconds
    }


@pytest.fixture
def performance_tracker() -> Generator[Dict[str, float], None, None]:
    """Track performance metrics during tests."""
    start_time = time.time()
    yield {"start_time": start_time}
    end_time = time.time()
    duration = end_time - start_time
    if duration > 10.0:  # Log slow tests
        print(f"\n⚠️  Slow test detected: {duration:.1f}s")


def pytest_runtest_setup(item: Any) -> None:
    """Setup for each test run."""
    # Skip slow tests unless explicitly requested
    if "slow" in item.keywords and not item.config.getoption(
        "--run-slow", default=False
    ):
        pytest.skip("Slow test skipped (use --run-slow to run)")


def pytest_addoption(parser: Any) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that take significant time",
    )
    parser.addoption(
        "--theoretical-only",
        action="store_true",
        default=False,
        help="Run only theoretical guarantee tests",
    )
    parser.addoption(
        "--empirical-only",
        action="store_true",
        default=False,
        help="Run only empirical validation tests",
    )
    parser.addoption(
        "--paper-validation",
        action="store_true",
        default=False,
        help="Run tests that directly validate paper claims",
    )


def pytest_collection_modifyitems(config: Any, items: List[Any]) -> None:
    """Modify test collection based on command line options."""
    if config.getoption("--theoretical-only"):
        selected_items = [item for item in items if "theoretical" in item.keywords]
        items[:] = selected_items

    elif config.getoption("--empirical-only"):
        selected_items = [item for item in items if "empirical" in item.keywords]
        items[:] = selected_items

    elif config.getoption("--paper-validation"):
        selected_items = [item for item in items if "paper_validation" in item.keywords]
        items[:] = selected_items


@pytest.fixture
def mock_arena_data() -> List[Dict[str, Any]]:
    """Generate mock Arena-Hard-like data for testing."""
    np.random.seed(42)
    n_samples = 100

    logs = []
    domains = ["reasoning", "coding", "creative", "factual"]

    for i in range(n_samples):
        domain = domains[i % len(domains)]

        # Simulate varying quality by domain
        domain_bias = {"reasoning": 6.5, "coding": 5.8, "creative": 7.2, "factual": 6.0}
        judge_score = np.clip(np.random.normal(domain_bias[domain], 1.5), 1.0, 10.0)

        # Oracle labels for 25% subset
        oracle_reward = None
        if i < n_samples // 4:
            oracle_reward = np.clip(judge_score / 10 + np.random.normal(0, 0.1), 0, 1)

        logs.append(
            {
                "context": f"{domain} task {i}",
                "response": f"Response to {domain} task",
                "logp": np.random.normal(-8, 2),
                "judge_raw": judge_score,
                "oracle_reward": oracle_reward,
                "domain": domain,
                "logp_target_all": {
                    "cot_policy": np.random.normal(-7.5, 2),
                    "few_shot": np.random.normal(-8.2, 2.1),
                },
            }
        )

    return logs


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: Any
) -> None:
    """Print summary of CJE test results."""
    print("\n" + "=" * 60)
    print("CJE Comprehensive Test Suite Summary")
    print("=" * 60)

    # Count tests by category
    theoretical_passed = 0
    empirical_passed = 0
    property_passed = 0
    integration_passed = 0

    for report in terminalreporter.stats.get("passed", []):
        if hasattr(report, "keywords"):
            if "theoretical" in report.keywords:
                theoretical_passed += 1
            if "empirical" in report.keywords:
                empirical_passed += 1
            if "property" in report.keywords:
                property_passed += 1
            if "integration" in report.keywords:
                integration_passed += 1

    print(f"✅ Theoretical Guarantees: {theoretical_passed} passed")
    print(f"✅ Empirical Validation: {empirical_passed} passed")
    print(f"✅ Property-Based Tests: {property_passed} passed")
    print(f"✅ Integration Tests: {integration_passed} passed")

    # Check for failures in critical categories
    failed_reports = terminalreporter.stats.get("failed", [])
    if failed_reports:
        print(f"\n❌ {len(failed_reports)} tests failed")
        for report in failed_reports:
            if hasattr(report, "keywords"):
                if "theoretical" in report.keywords:
                    print(
                        f"   🔴 CRITICAL: Theoretical guarantee failed: {report.nodeid}"
                    )
                elif "paper_validation" in report.keywords:
                    print(
                        f"   🟡 WARNING: Paper claim validation failed: {report.nodeid}"
                    )

    # Overall assessment
    if exitstatus == 0:
        print("\n🎉 All CJE tests passed! Repository ready for release.")
    else:
        print(f"\n⚠️  Some tests failed (exit code: {exitstatus})")
        print("Please review failures before release.")

    print("=" * 60)
