"""
Mock dataset implementation for testing without external dependencies.

This module provides a mock dataset that can simulate CJEDataset behavior
without requiring actual data files or external datasets.
"""

from typing import Iterable, List, Dict, Any, Optional, Union, Type
from pathlib import Path

# Import base classes
try:
    from ...data.base import CJEDataset
    from ...data.schema import CJESample

    # Use real classes if available
    DatasetBase = CJEDataset
    SampleClass = CJESample
except ImportError:
    # Fallback for testing isolation
    class FallbackCJEDataset:
        def itersamples(self) -> Iterable[Any]:
            return iter([])

    class FallbackCJESample:
        def __init__(
            self,
            uid: str,
            context: str,
            response: str,
            y_true: Optional[Any] = None,
            logp: Optional[float] = None,
            meta: Optional[Dict[str, Any]] = None,
        ) -> None:
            self.uid = uid
            self.context = context
            self.response = response
            self.y_true = y_true
            self.logp = logp
            self.meta = meta or {}

    # Use fallback classes
    DatasetBase = FallbackCJEDataset  # type: ignore
    SampleClass = FallbackCJESample  # type: ignore


class MockDataset(DatasetBase):  # type: ignore
    """
    Mock implementation of CJEDataset for testing.

    Generates realistic test data without requiring external files or APIs.
    """

    def __init__(self, name: str = "mock_dataset", split: str = "test", size: int = 10):
        """
        Initialize mock dataset.

        Args:
            name: Dataset name (used to determine data characteristics)
            split: Dataset split (affects which samples are generated)
            size: Number of samples to generate
        """
        self.name = name
        self.split = split
        self.size = size

        # Generate data based on name and split
        self._samples = self._generate_samples()

    def _generate_samples(self) -> List[SampleClass]:
        """Generate mock samples based on dataset configuration."""
        samples = []

        # Sample contexts for different dataset types
        if "summeval" in self.name.lower():
            contexts = self._generate_summarization_contexts()
        else:
            contexts = self._generate_general_contexts()

        # Generate samples
        for i in range(self.size):
            context = contexts[i % len(contexts)]

            # Generate corresponding response
            response = self._generate_response_for_context(context, i)

            # Generate realistic log probability
            logp = self._generate_logp(context, response, i)

            # Generate ground truth if appropriate
            y_true = self._generate_ground_truth(context, response, i)

            sample = SampleClass(
                uid=f"{self.name}_{self.split}_{i:04d}",
                context=context,
                response=response,
                y_true=y_true,
                logp=logp,
                meta={"mock_dataset": True, "split": self.split, "index": i},
            )
            samples.append(sample)

        return samples

    def _generate_summarization_contexts(self) -> List[str]:
        """Generate contexts for summarization tasks."""
        return [
            "Summarize the following article: Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed for every task.",
            "Please provide a summary: Deep learning neural networks have revolutionized computer vision by automatically learning hierarchical feature representations from raw pixel data.",
            "Summarize this text: Natural language processing has advanced significantly with the introduction of transformer models, which use attention mechanisms to process text more effectively.",
            "Create a summary: Reinforcement learning enables agents to learn optimal behaviors through trial and error interactions with their environment, maximizing cumulative rewards.",
            "Summarize: Computer vision applications range from medical image analysis to autonomous vehicles, requiring robust algorithms that can handle diverse visual scenarios.",
        ]

    def _generate_general_contexts(self) -> List[str]:
        """Generate general-purpose contexts."""
        return [
            "What is the capital of France?",
            "Explain the process of photosynthesis.",
            "How do neural networks work?",
            "What are the benefits of exercise?",
            "Describe the water cycle.",
            "What is the theory of relativity?",
            "How do vaccines work?",
            "What causes climate change?",
            "Explain how the internet works.",
            "What is quantum mechanics?",
        ]

    def _generate_response_for_context(self, context: str, index: int) -> str:
        """Generate appropriate response for a given context."""
        import hashlib

        # Use context hash for deterministic but varied responses
        context_hash = int(hashlib.md5(context.encode()).hexdigest()[:8], 16)
        response_type = context_hash % 3

        if "summarize" in context.lower() or "summary" in context.lower():
            # Summarization response
            if response_type == 0:
                return "This text discusses the key concepts and principles behind the topic, highlighting important aspects and their implications."
            elif response_type == 1:
                return "The main points include several fundamental ideas that are essential for understanding the subject matter."
            else:
                return "In summary, the content covers critical information about the topic and its relevance to current applications."

        elif "?" in context:
            # Question answering response
            if response_type == 0:
                return "This is a comprehensive answer that addresses the key aspects of your question with relevant details and examples."
            elif response_type == 1:
                return "The answer involves multiple factors and considerations that are important to understand in this context."
            else:
                return "To address your question, there are several important points to consider that provide a complete picture."

        else:
            # General response
            if response_type == 0:
                return "This topic involves several important concepts that are fundamental to understanding the broader subject area."
            elif response_type == 1:
                return "There are multiple aspects to consider when examining this subject, each contributing to the overall understanding."
            else:
                return "The key principles underlying this topic provide valuable insights into its practical applications and significance."

    def _generate_logp(self, context: str, response: str, index: int) -> float:
        """Generate realistic log probability for the response."""
        import hashlib

        # Base log probability on response characteristics
        response_length = len(response.split())
        context_length = len(context.split())

        # Longer responses generally have lower probability
        base_logp = -2.0 - (response_length * 0.1)

        # Add deterministic noise based on content
        content_hash = int(
            hashlib.md5(f"{context}{response}".encode()).hexdigest()[:8], 16
        )
        noise = (content_hash % 1000) / 1000.0  # 0-1 range
        noise_adjustment = (noise - 0.5) * 3.0  # -1.5 to 1.5 range

        final_logp = base_logp + noise_adjustment

        # Ensure reasonable bounds
        return max(-50.0, min(-1.0, final_logp))

    def _generate_ground_truth(
        self, context: str, response: str, index: int
    ) -> Optional[float]:
        """Generate ground truth labels for some samples."""
        # Only generate ground truth for some samples (simulating partial labeling)
        if index % 3 == 0:  # Every third sample has ground truth
            # Base quality on response characteristics
            response_length = len(response.split())

            # Longer, more detailed responses get higher scores
            if response_length > 20:
                base_score = 0.8
            elif response_length > 10:
                base_score = 0.6
            else:
                base_score = 0.4

            # Add deterministic variation
            import hashlib

            hash_val = int(hashlib.md5(response.encode()).hexdigest()[:8], 16)
            variation = (hash_val % 100) / 1000.0  # Small variation

            final_score = base_score + variation
            return max(0.0, min(1.0, final_score))
        else:
            return None

    def itersamples(self) -> Iterable[SampleClass]:
        """Iterate over mock samples."""
        for sample in self._samples:
            yield sample

    def __len__(self) -> int:
        """Get number of samples."""
        return len(self._samples)

    def __str__(self) -> str:
        return (
            f"MockDataset(name='{self.name}', split='{self.split}', size={self.size})"
        )

    def __repr__(self) -> str:
        return (
            f"MockDataset(name='{self.name}', split='{self.split}', size={self.size})"
        )

    @classmethod
    def download(
        cls, cache_dir: Optional[str] = None, split: str = "train"
    ) -> "MockDataset":
        """Create mock dataset (simulates download)."""
        return cls(name="mock_downloaded_dataset", split=split, size=15)


# Convenience function for creating mock datasets
def create_mock_dataset(
    dataset_type: str = "general",
    split: str = "test",
    size: int = 10,
    with_ground_truth: bool = True,
) -> "MockDataset":
    """
    Create a mock dataset with specific characteristics.

    Args:
        dataset_type: Type of dataset ("summeval", "general")
        split: Dataset split
        size: Number of samples
        with_ground_truth: Whether to include ground truth labels

    Returns:
        MockDataset instance
    """
    dataset_name = f"mock_{dataset_type}_dataset"
    dataset = MockDataset(name=dataset_name, split=split, size=size)

    # Remove ground truth if not requested
    if not with_ground_truth:
        for sample in dataset._samples:
            sample.y_true = None

    return dataset
