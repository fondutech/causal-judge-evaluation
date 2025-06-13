"""
Data fixtures for CJE testing.

Provides common test datasets, contexts, responses, and ground truth labels
for testing CJE components across different scenarios.
"""

import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pytest


# Sample contexts covering different types of tasks
sample_contexts = [
    # Question answering
    "What is machine learning?",
    "How do neural networks work?",
    "Explain the difference between supervised and unsupervised learning.",
    "What are the main challenges in natural language processing?",
    # Creative writing
    "Write a short story about a robot discovering emotions.",
    "Compose a poem about the changing seasons.",
    "Create a dialogue between a scientist and an artist.",
    # Technical explanations
    "Explain how gradient descent works in machine learning.",
    "Describe the architecture of a transformer model.",
    "What is the attention mechanism in deep learning?",
    # Problem solving
    "How would you design a recommendation system?",
    "What approach would you take to detect fraud in financial transactions?",
    "Suggest ways to improve customer satisfaction in e-commerce.",
    # Analysis and reasoning
    "Compare the advantages and disadvantages of cloud computing.",
    "Analyze the impact of social media on modern communication.",
    "Discuss the ethical considerations of artificial intelligence.",
]

# Sample responses of varying quality for testing judge behavior
sample_responses = {
    "high_quality": [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns.",
        "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information through weighted connections, learning to recognize patterns through training.",
        "Supervised learning uses labeled training data to learn mappings from inputs to outputs, while unsupervised learning finds hidden patterns in unlabeled data without specific target outputs.",
    ],
    "medium_quality": [
        "Machine learning is when computers learn things automatically. It uses data to make predictions and decisions without people telling it exactly what to do.",
        "Neural networks are like artificial brains with connected parts that process information. They learn by adjusting connections between parts.",
        "Supervised learning has answers provided, unsupervised learning finds patterns without answers given.",
    ],
    "low_quality": [
        "Machine learning is computers learning stuff. It's AI related and uses data and algorithms and makes predictions.",
        "Neural networks are networks of neurons that are artificial and connected and process things and learn stuff.",
        "Supervised has supervision, unsupervised doesn't have supervision. One has labels one doesn't.",
    ],
}

# Sample ground truth scores (0-1 scale)
sample_ground_truth = {
    "high_quality": [0.9, 0.85, 0.88, 0.92, 0.87],
    "medium_quality": [0.6, 0.65, 0.58, 0.62, 0.61],
    "low_quality": [0.2, 0.25, 0.18, 0.22, 0.28],
}


def create_test_dataset(
    size: int = 20,
    quality_distribution: Optional[Dict[str, float]] = None,
    scenario: str = "2",
    include_ground_truth: bool = True,
    file_path: Optional[str] = None,
) -> Union[List[Dict[str, Any]], str]:
    """
    Create a test dataset for CJE testing.

    Args:
        size: Number of samples to generate
        quality_distribution: Distribution of quality levels (e.g., {"high": 0.3, "medium": 0.5, "low": 0.2})
        scenario: CJE scenario type ("1", "2", or "3")
        include_ground_truth: Whether to include ground truth labels
        file_path: If provided, write dataset to this file and return path

    Returns:
        List of data samples or file path if file_path is provided
    """
    if quality_distribution is None:
        quality_distribution = {
            "high_quality": 0.4,
            "medium_quality": 0.4,
            "low_quality": 0.2,
        }

    # Validate quality distribution
    if abs(sum(quality_distribution.values()) - 1.0) > 0.01:
        raise ValueError("Quality distribution must sum to 1.0")

    dataset = []
    contexts_cycle = sample_contexts * (size // len(sample_contexts) + 1)

    for i in range(size):
        context = contexts_cycle[i]
        sample: Dict[str, Any] = {"uid": f"test_{i:03d}", "context": context}

        # Determine quality level for this sample
        import random

        random.seed(42 + i)  # Deterministic but varied
        quality_level = random.choices(
            list(quality_distribution.keys()),
            weights=list(quality_distribution.values()),
        )[0]

        # Initialize logp for all scenarios
        logp: float = 0.0

        if scenario in ["2", "3"]:
            # Include response
            responses = sample_responses[quality_level]
            response_idx = i % len(responses)
            sample["response"] = responses[response_idx]

            # Add mock log probability (more negative for lower quality)
            if quality_level == "high_quality":
                logp = random.uniform(-8, -5)
            elif quality_level == "medium_quality":
                logp = random.uniform(-15, -10)
            else:  # low_quality
                logp = random.uniform(-25, -20)
            sample["logp"] = logp

        if include_ground_truth and scenario == "2":
            # Add ground truth score
            gt_scores = sample_ground_truth[quality_level]
            gt_idx = i % len(gt_scores)
            sample["y_true"] = gt_scores[gt_idx]

        if scenario == "3":
            # Pre-computed target samples
            target_samples: List[Dict[str, Any]] = []
            for temp in [0.1, 0.7, 1.2]:
                # Generate different responses for different temperatures
                temp_responses = sample_responses[quality_level]
                temp_response = temp_responses[
                    (i + int(temp * 10)) % len(temp_responses)
                ]

                # Adjust response based on temperature
                if temp > 1.0:
                    temp_response = (
                        temp_response
                        + " This approach encourages creative exploration and innovative thinking."
                    )
                elif temp < 0.3:
                    temp_response = (
                        temp_response.split(".")[0] + "."
                    )  # More conservative

                temp_logp = (
                    logp + random.uniform(-2, 2) * temp
                )  # Temperature affects logp
                target_samples.append({"response": temp_response, "logp": temp_logp})
            sample["target_samples"] = target_samples

        dataset.append(sample)

    if file_path:
        # Write to file
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for sample in dataset:
                f.write(json.dumps(sample) + "\n")
        return str(path)
    else:
        return dataset


def scenario_1_data(size: int = 10) -> List[Dict[str, Any]]:
    """Create Scenario 1 test data (context only)."""
    result = create_test_dataset(size=size, scenario="1", include_ground_truth=False)
    # create_test_dataset returns list when file_path is None
    return result  # type: ignore


def scenario_2_data(size: int = 10) -> List[Dict[str, Any]]:
    """Create Scenario 2 test data (complete logs with ground truth)."""
    result = create_test_dataset(size=size, scenario="2", include_ground_truth=True)
    # create_test_dataset returns list when file_path is None
    return result  # type: ignore


def scenario_3_data(size: int = 10) -> List[Dict[str, Any]]:
    """Create Scenario 3 test data (pre-computed policy data)."""
    result = create_test_dataset(size=size, scenario="3", include_ground_truth=False)
    # create_test_dataset returns list when file_path is None
    return result  # type: ignore


# Pytest fixtures for easy use in tests
@pytest.fixture
def sample_scenario_1_data() -> List[Dict[str, Any]]:
    """Pytest fixture for Scenario 1 data."""
    return scenario_1_data(size=5)


@pytest.fixture
def sample_scenario_2_data() -> List[Dict[str, Any]]:
    """Pytest fixture for Scenario 2 data."""
    return scenario_2_data(size=5)


@pytest.fixture
def sample_scenario_3_data() -> List[Dict[str, Any]]:
    """Pytest fixture for Scenario 3 data."""
    return scenario_3_data(size=5)


@pytest.fixture
def temp_dataset_file() -> Any:
    """Pytest fixture for temporary dataset file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Create small test dataset
        data = scenario_2_data(size=3)
        for sample in data:
            f.write(json.dumps(sample) + "\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def varied_quality_dataset() -> List[Dict[str, Any]]:
    """Pytest fixture for dataset with varied quality levels."""
    result = create_test_dataset(
        size=15,
        quality_distribution={
            "high_quality": 0.2,
            "medium_quality": 0.5,
            "low_quality": 0.3,
        },
        scenario="2",
    )
    # create_test_dataset returns list when file_path is None
    return result  # type: ignore


# Special datasets for edge case testing
def empty_responses_data(size: int = 5) -> List[Dict[str, Any]]:
    """Create data with empty responses for edge case testing."""
    data = []
    for i in range(size):
        data.append(
            {
                "uid": f"empty_{i}",
                "context": sample_contexts[i % len(sample_contexts)],
                "response": "",  # Empty response
                "logp": -100.0,  # Very low probability
                "y_true": 0.0,  # Low score
            }
        )
    return data


def very_long_responses_data(size: int = 3) -> List[Dict[str, Any]]:
    """Create data with very long responses for testing limits."""
    long_response = " ".join(sample_responses["high_quality"]) * 5  # Very long

    data = []
    for i in range(size):
        data.append(
            {
                "uid": f"long_{i}",
                "context": sample_contexts[i],
                "response": long_response,
                "logp": -50.0,  # Long responses typically have lower probability
                "y_true": 0.6,  # Medium quality due to length
            }
        )
    return data


def mixed_format_data(size: int = 5) -> List[Dict[str, Any]]:
    """Create data with mixed field formats for robustness testing."""
    data = []
    for i in range(size):
        sample: Dict[str, Any] = {
            "uid": f"mixed_{i}",
            "context": sample_contexts[i % len(sample_contexts)],
            "response": sample_responses["medium_quality"][
                i % len(sample_responses["medium_quality"])
            ],
        }

        # Vary the format of numeric fields
        if i % 3 == 0:
            sample["logp"] = -10.5  # Float
            sample["y_true"] = 0.7  # Float
        elif i % 3 == 1:
            sample["logp"] = -15  # Int
            sample["y_true"] = 1  # Int
        else:
            sample["logp"] = "-12.3"  # String (should be converted)
            sample["y_true"] = "0.6"  # String (should be converted)

        data.append(sample)
    return data


# Export commonly used datasets
__all__ = [
    "sample_contexts",
    "sample_responses",
    "sample_ground_truth",
    "create_test_dataset",
    "scenario_1_data",
    "scenario_2_data",
    "scenario_3_data",
    "empty_responses_data",
    "very_long_responses_data",
    "mixed_format_data",
]
