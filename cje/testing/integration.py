"""
Integration utilities for CJE testing infrastructure.

This module provides functions to enable testing mode across CJE components,
create mock pipelines, and integrate mock implementations seamlessly.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Type
from contextlib import contextmanager
import logging

from .mocks.policy_runners import MockPolicyRunner, MockAPIPolicyRunner, MockModelConfig
from .mocks.judges import MockJudge, MockAPIJudge, MockLocalJudge, create_mock_judge
from .mocks.multi_target_sampler import (
    MockMultiTargetSampler,
    create_mock_multi_sampler,
)
from .fixtures.data import create_test_dataset
from .fixtures.configs import basic_config, validate_config_structure

logger = logging.getLogger(__name__)

# Global state for testing mode
_testing_mode_enabled = False
_original_factories: Dict[str, Any] = {}


def enable_testing_mode(
    mock_policy_runners: bool = True,
    mock_judges: bool = True,
    mock_datasets: bool = True,
    suppress_warnings: bool = True,
) -> None:
    """
    Enable testing mode across CJE components.

    This function patches CJE's factories and loaders to use mock implementations
    instead of real models and APIs.

    Args:
        mock_policy_runners: Replace PolicyRunner and APIPolicyRunner with mocks
        mock_judges: Replace judge implementations with mocks
        mock_datasets: Replace dataset loading with mock datasets
        suppress_warnings: Suppress non-critical warnings during testing
    """
    global _testing_mode_enabled, _original_factories

    if _testing_mode_enabled:
        logger.warning("Testing mode already enabled")
        return

    logger.info("Enabling CJE testing mode")

    # Store original implementations for restoration
    _original_factories.clear()

    try:
        # Mock policy runners
        if mock_policy_runners:
            _patch_policy_runners()

        # Mock judges
        if mock_judges:
            _patch_judges()

        # Mock datasets
        if mock_datasets:
            _patch_datasets()

        # Suppress warnings if requested
        if suppress_warnings:
            _suppress_testing_warnings()

        _testing_mode_enabled = True
        logger.info("✅ CJE testing mode enabled successfully")

    except Exception as e:
        logger.error(f"Failed to enable testing mode: {e}")
        # Attempt to restore if partial failure
        disable_testing_mode()
        raise


def disable_testing_mode() -> None:
    """
    Disable testing mode and restore original implementations.
    """
    global _testing_mode_enabled, _original_factories

    if not _testing_mode_enabled:
        return

    logger.info("Disabling CJE testing mode")

    try:
        # Restore original implementations
        _restore_original_implementations()

        _testing_mode_enabled = False
        _original_factories.clear()

        logger.info("✅ CJE testing mode disabled successfully")

    except Exception as e:
        logger.error(f"Error disabling testing mode: {e}")


@contextmanager
def testing_mode(
    mock_policy_runners: bool = True,
    mock_judges: bool = True,
    mock_datasets: bool = True,
    suppress_warnings: bool = True,
) -> Any:
    """
    Context manager for temporary testing mode.

    Usage:
        with testing_mode():
            # CJE components will use mocks
            result = run_cje_pipeline(config)
        # Original implementations restored
    """
    was_enabled = _testing_mode_enabled

    if not was_enabled:
        enable_testing_mode(
            mock_policy_runners, mock_judges, mock_datasets, suppress_warnings
        )

    try:
        yield
    finally:
        if not was_enabled:
            disable_testing_mode()


def _patch_policy_runners() -> None:
    """Patch policy runner factories to use mocks."""
    try:
        # Patch PolicyRunner
        from ..loggers import policy

        _original_factories["PolicyRunner"] = policy.PolicyRunner
        setattr(policy, "PolicyRunner", MockPolicyRunner)

        # Patch APIPolicyRunner
        from ..loggers import api_policy

        _original_factories["APIPolicyRunner"] = api_policy.APIPolicyRunner
        setattr(api_policy, "APIPolicyRunner", MockAPIPolicyRunner)

        # Patch adapter classes to use mock implementations
        _patch_adapters()

        logger.debug("Policy runners and adapters patched with mocks")

    except ImportError as e:
        logger.warning(f"Could not patch policy runners: {e}")


def _patch_adapters() -> None:
    """Patch adapter classes to use mock implementations."""
    try:
        from ..loggers import adapters

        # Store original adapter classes
        _original_factories["OpenAIAdapter"] = adapters.OpenAIAdapter
        _original_factories["AnthropicAdapter"] = adapters.AnthropicAdapter
        _original_factories["GeminiAdapter"] = adapters.GeminiAdapter

        # Create mock adapter classes that don't require real API clients
        class MockOpenAIAdapter:
            def __init__(
                self,
                model_name: str,
                client: Optional[Any] = None,
                system_prompt: Optional[str] = None,
                user_message_template: str = "{context}",
            ) -> None:
                self.model_name = model_name
                self.client = "mock_openai_client"  # Mock client
                self.system_prompt = system_prompt
                self.user_message_template = user_message_template

            def _parse_context(self, context: str) -> Any:
                # Use the same conversation parsing logic
                from ..loggers.conversation_utils import parse_context

                return parse_context(
                    context, self.system_prompt, self.user_message_template
                )

        class MockAnthropicAdapter:
            def __init__(
                self,
                model_name: str,
                client: Optional[Any] = None,
                system_prompt: Optional[str] = None,
                user_message_template: str = "{context}",
            ) -> None:
                self.model_name = model_name
                self.client = "mock_anthropic_client"  # Mock client
                self.system_prompt = system_prompt
                self.user_message_template = user_message_template

            def _parse_context(self, context: str) -> Any:
                from ..loggers.conversation_utils import parse_context

                return parse_context(
                    context, self.system_prompt, self.user_message_template
                )

        class MockGeminiAdapter:
            def __init__(
                self,
                model_name: str,
                client: Optional[Any] = None,
                system_prompt: Optional[str] = None,
                user_message_template: str = "{context}",
            ) -> None:
                self.model_name = model_name
                self.client = "mock_gemini_client"  # Mock client
                self.system_prompt = system_prompt
                self.user_message_template = user_message_template

            def _parse_context(self, context: str) -> Any:
                from ..loggers.conversation_utils import parse_context

                return parse_context(
                    context, self.system_prompt, self.user_message_template
                )

            def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
                """Convert messages to text format for Gemini."""
                text_parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        text_parts.append(f"System: {content}")
                    elif role == "user":
                        text_parts.append(f"Human: {content}")
                    elif role == "assistant":
                        text_parts.append(f"AI: {content}")
                return "\n".join(text_parts)

        # Replace adapter classes with mock versions
        setattr(adapters, "OpenAIAdapter", MockOpenAIAdapter)
        setattr(adapters, "AnthropicAdapter", MockAnthropicAdapter)
        setattr(adapters, "GeminiAdapter", MockGeminiAdapter)

        logger.debug("Adapter classes patched with mocks")

    except ImportError as e:
        logger.warning(f"Could not patch adapters: {e}")


def _patch_judges() -> None:
    """Patch judge factories to use mocks."""
    try:
        # Patch JudgeFactory
        from ..judge import factory

        _original_factories["JudgeFactory.create"] = factory.JudgeFactory.create

        def mock_judge_factory(
            judge_type: str,
            config: Union[Dict[str, Any], Any],
            use_cache: bool = True,
            cache_size: int = 1000,
        ) -> Any:
            """Mock judge factory that returns appropriate mock judges."""
            if isinstance(config, dict):
                config_dict = config
            else:
                # Convert config object to dict
                config_dict = {}
                for attr in dir(config):
                    if not attr.startswith("_"):
                        config_dict[attr] = getattr(config, attr)

            mock_judge = create_mock_judge(judge_type, config_dict)

            if use_cache:
                from ..judge.base import CachedJudge

                return CachedJudge(mock_judge, cache_size)
            else:
                return mock_judge

        # Use setattr to avoid method assignment error
        setattr(factory.JudgeFactory, "create", staticmethod(mock_judge_factory))

        logger.debug("Judge factory patched with mocks")

    except ImportError as e:
        logger.warning(f"Could not patch judge factory: {e}")


def _patch_datasets() -> None:
    """Patch dataset loading to use mock datasets."""
    try:
        # Import the load_dataset function directly
        from ..data import load_dataset

        _original_factories["load_dataset"] = load_dataset

        def mock_load_dataset(
            name: Union[str, Path], split: Optional[str] = None
        ) -> Any:
            """Mock dataset loader that returns mock datasets."""
            # Check if it's a request for mock dataset
            if isinstance(name, str) and "mock" in name.lower():
                # Create a mock dataset - handle Optional[str] for split
                from .mocks.dataset import MockDataset

                split_str = split if split is not None else "test"
                return MockDataset(name, split_str)
            else:
                # Use original loader for real datasets
                return _original_factories["load_dataset"](name, split)

        # Patch the function in the data module
        import cje.data as data_module

        data_module.load_dataset = mock_load_dataset

        logger.debug("Dataset loading patched with mocks")

    except ImportError as e:
        logger.warning(f"Could not patch dataset loading: {e}")
    except Exception as e:
        logger.warning(f"Error patching datasets: {e}")


def _suppress_testing_warnings() -> None:
    """Suppress non-critical warnings during testing."""
    # Suppress specific warnings that are not relevant during testing
    import warnings

    # Suppress model loading warnings
    warnings.filterwarnings("ignore", message=".*model was not found.*")
    warnings.filterwarnings("ignore", message=".*API key not found.*")
    warnings.filterwarnings("ignore", message=".*rate limit.*")

    # Suppress PyTorch warnings in testing
    warnings.filterwarnings("ignore", message=".*PyTorch version.*")


def _restore_original_implementations() -> None:
    """Restore original implementations from stored factories."""
    for name, original in _original_factories.items():
        try:
            if name == "PolicyRunner":
                from ..loggers import policy

                setattr(policy, "PolicyRunner", original)
            elif name == "APIPolicyRunner":
                from ..loggers import api_policy

                setattr(api_policy, "APIPolicyRunner", original)
            elif name == "OpenAIAdapter":
                from ..loggers import adapters

                setattr(adapters, "OpenAIAdapter", original)
            elif name == "AnthropicAdapter":
                from ..loggers import adapters

                setattr(adapters, "AnthropicAdapter", original)
            elif name == "GeminiAdapter":
                from ..loggers import adapters

                setattr(adapters, "GeminiAdapter", original)
            elif name == "JudgeFactory.create":
                from ..judge import factory

                # Use setattr to restore as well
                setattr(factory.JudgeFactory, "create", original)
            elif name == "load_dataset":
                import cje.data as data_module

                data_module.load_dataset = original

        except Exception as e:
            logger.warning(f"Could not restore {name}: {e}")


def create_mock_pipeline(
    scenario: str = "basic",
    size: int = 5,
    config_overrides: Optional[Dict[str, Any]] = None,
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a complete mock pipeline for testing.

    Args:
        scenario: Type of scenario ("basic", "multi_policy", "temperature_sweep")
        size: Size of test dataset
        config_overrides: Additional config parameters
        work_dir: Working directory (creates temp if None)

    Returns:
        Dictionary with pipeline components and configuration
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp()

    # Create test data
    if scenario == "basic":
        from .fixtures.data import scenario_2_data

        data = scenario_2_data(size=size)
        config = basic_config(work_dir=work_dir, dataset_size=size)
    elif scenario == "multi_policy":
        from .fixtures.configs import multi_policy_config
        from .fixtures.data import scenario_2_data

        data = scenario_2_data(size=size)
        config = multi_policy_config(work_dir=work_dir, dataset_size=size)
    elif scenario == "temperature_sweep":
        from .fixtures.configs import temperature_sweep_config
        from .fixtures.data import scenario_2_data

        data = scenario_2_data(size=size)
        config = temperature_sweep_config(work_dir=work_dir)
        config["dataset"]["sample_limit"] = size
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Apply config overrides
    if config_overrides:
        config = _deep_merge_configs(config, config_overrides)

    # Create mock components
    components: Dict[str, Any] = {}

    # Create mock policy runners
    logging_policy_config = config["logging_policy"]
    if logging_policy_config.get("provider") == "hf":
        components["logging_runner"] = MockPolicyRunner(
            model_name=logging_policy_config["model_name"],
            temperature=logging_policy_config.get("temperature", 0.7),
            max_new_tokens=logging_policy_config.get("max_new_tokens", 50),
        )
    else:
        components["logging_runner"] = MockAPIPolicyRunner(
            provider=logging_policy_config["provider"],
            model_name=logging_policy_config["model_name"],
            temperature=logging_policy_config.get("temperature", 0.7),
            max_new_tokens=logging_policy_config.get("max_new_tokens", 50),
        )

    # Create mock target sampler
    policy_configs = []
    for target_policy in config["target_policies"]:
        policy_config = {
            "model_name": target_policy["model_name"],
            "temperature": target_policy.get("temperature", 0.5),
            "max_new_tokens": target_policy.get("max_new_tokens", 50),
        }
        if "provider" in target_policy:
            policy_config["provider"] = target_policy["provider"]
        policy_configs.append(policy_config)

    components["target_sampler"] = create_mock_multi_sampler(policy_configs)

    # Create mock judge
    judge_config = config["judge"]
    components["judge"] = create_mock_judge(
        judge_type=judge_config["provider"], config=judge_config
    )

    # Write test data to file
    data_file = Path(work_dir) / "test_data.jsonl"
    data_file.parent.mkdir(parents=True, exist_ok=True)

    with open(data_file, "w") as f:
        for sample in data:
            import json

            f.write(json.dumps(sample) + "\n")

    # Update config to point to data file
    config["dataset"]["name"] = str(data_file)

    return {
        "config": config,
        "data": data,
        "components": components,
        "work_dir": work_dir,
        "data_file": str(data_file),
    }


def _deep_merge_configs(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep merge configuration dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def validate_mock_pipeline(pipeline: Dict[str, Any]) -> List[str]:
    """
    Validate that a mock pipeline is properly configured.

    Args:
        pipeline: Pipeline created by create_mock_pipeline

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required keys
    required_keys = ["config", "data", "components", "work_dir", "data_file"]
    for key in required_keys:
        if key not in pipeline:
            errors.append(f"Missing pipeline key: {key}")

    # Validate config structure
    if "config" in pipeline:
        config_errors = validate_config_structure(pipeline["config"])
        errors.extend(config_errors)

    # Check components
    if "components" in pipeline:
        components = pipeline["components"]
        required_components = ["logging_runner", "target_sampler", "judge"]
        for component in required_components:
            if component not in components:
                errors.append(f"Missing pipeline component: {component}")

    # Check data file exists
    if "data_file" in pipeline:
        data_file = Path(pipeline["data_file"])
        if not data_file.exists():
            errors.append(f"Data file does not exist: {data_file}")

    return errors


def run_mock_pipeline_test(
    pipeline: Dict[str, Any], expected_outcomes: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run a test using a mock pipeline.

    Args:
        pipeline: Mock pipeline created by create_mock_pipeline
        expected_outcomes: Expected test outcomes for validation

    Returns:
        Test results dictionary
    """
    import time

    start_time = time.time()

    results: Dict[str, Any] = {
        "success": False,
        "runtime_seconds": 0.0,
        "errors": [],
        "warnings": [],
        "outputs": {},
    }

    try:
        # Validate pipeline
        validation_errors = validate_mock_pipeline(pipeline)
        if validation_errors:
            results["errors"].extend(validation_errors)
            return results

        # Test components individually
        components = pipeline["components"]
        data = pipeline["data"]

        # Test logging runner
        if "logging_runner" in components:
            runner = components["logging_runner"]
            test_context = data[0]["context"] if data else "Test context"

            try:
                # Test generation
                gen_results = runner.generate_with_logp([test_context])
                if not gen_results:
                    results["warnings"].append("Logging runner generated no results")
                else:
                    results["outputs"]["logging_generation"] = gen_results[0]

                # Test log probability
                test_response = gen_results[0][0] if gen_results else "Test response"
                logp = runner.log_prob(test_context, test_response)
                results["outputs"]["logging_logp"] = logp

            except Exception as e:
                results["errors"].append(f"Logging runner error: {e}")

        # Test target sampler
        if "target_sampler" in components:
            sampler = components["target_sampler"]
            test_context = data[0]["context"] if data else "Test context"
            test_response = (
                data[0].get("response", "Test response") if data else "Test response"
            )

            try:
                # Test log probability computation
                logps = sampler.logp_many(test_context, test_response)
                results["outputs"]["target_logps"] = logps

                # Test sampling
                samples = sampler.sample_many(test_context, n=2)
                results["outputs"]["target_samples"] = samples

            except Exception as e:
                results["errors"].append(f"Target sampler error: {e}")

        # Test judge
        if "judge" in components:
            judge = components["judge"]
            test_context = data[0]["context"] if data else "Test context"
            test_response = (
                data[0].get("response", "Test response") if data else "Test response"
            )

            try:
                score = judge.score(test_context, test_response)
                results["outputs"]["judge_score"] = score

                # Test batch scoring
                batch_samples = [{"context": test_context, "response": test_response}]
                batch_scores = judge.score_batch(batch_samples)
                results["outputs"]["judge_batch_scores"] = batch_scores

            except Exception as e:
                results["errors"].append(f"Judge error: {e}")

        # Mark success if no errors
        if not results["errors"]:
            results["success"] = True

    except Exception as e:
        results["errors"].append(f"Pipeline test error: {e}")

    finally:
        results["runtime_seconds"] = time.time() - start_time

    return results


# Export public API
__all__ = [
    "enable_testing_mode",
    "disable_testing_mode",
    "testing_mode",
    "create_mock_pipeline",
    "validate_mock_pipeline",
    "run_mock_pipeline_test",
]
