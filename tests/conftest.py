"""
Simplified pytest configuration for CJE test suite.

This module configures pytest with a simplified categorization:
- unit: Fast unit tests (default, always run)
- integration: End-to-end integration tests
- slow: Tests that take significant time
"""

import pytest
import numpy as np
import time
from typing import Any, Generator, List, Dict
import warnings


def pytest_configure(config: Any) -> None:
    """Configure pytest with simplified markers."""
    config.addinivalue_line(
        "markers",
        "unit: Fast unit tests that test individual components (default)",
    )
    config.addinivalue_line(
        "markers",
        "integration: End-to-end integration tests that test the full pipeline",
    )
    config.addinivalue_line(
        "markers",
        "slow: Tests that take more than a few seconds to run",
    )


def pytest_addoption(parser: Any) -> None:
    """Add simplified command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (default: skip)",
    )
    parser.addoption(
        "--integration-only",
        action="store_true",
        default=False,
        help="Run only integration tests",
    )
    parser.addoption(
        "--unit-only",
        action="store_true",
        default=False,
        help="Run only unit tests",
    )


def pytest_runtest_setup(item: Any) -> None:
    """Setup for each test run."""
    # Skip slow tests unless explicitly requested
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("Slow test skipped (use --run-slow to run)")


def pytest_collection_modifyitems(config: Any, items: List[Any]) -> None:
    """Filter tests based on command line options."""
    if config.getoption("--unit-only"):
        # Run only tests that are NOT marked as integration or slow
        selected_items = [
            item
            for item in items
            if "integration" not in item.keywords and "slow" not in item.keywords
        ]
        items[:] = selected_items

    elif config.getoption("--integration-only"):
        # Run only integration tests
        selected_items = [item for item in items if "integration" in item.keywords]
        items[:] = selected_items


# Common fixtures


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
    """Print simplified test summary."""
    print("\n" + "=" * 60)
    print("CJE Test Summary")
    print("=" * 60)

    # Count tests by category
    unit_count = 0
    integration_count = 0
    slow_count = 0

    all_reports = []
    for status in ["passed", "failed", "skipped"]:
        all_reports.extend(terminalreporter.stats.get(status, []))

    for report in all_reports:
        if hasattr(report, "keywords"):
            if "slow" in report.keywords:
                slow_count += 1
            elif "integration" in report.keywords:
                integration_count += 1
            else:
                unit_count += 1

    print(f"✅ Unit Tests: {unit_count}")
    print(f"✅ Integration Tests: {integration_count}")
    print(f"✅ Slow Tests: {slow_count}")

    # Report failures
    failed_reports = terminalreporter.stats.get("failed", [])
    if failed_reports:
        print(f"\n❌ {len(failed_reports)} tests failed")

    # Overall assessment
    if exitstatus == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  Some tests failed (exit code: {exitstatus})")

    print("=" * 60)
