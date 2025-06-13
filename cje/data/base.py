"""Base classes for CJE datasets."""

from abc import ABC, abstractmethod
from typing import Iterable, Optional


from .schema import CJESample


class CJEDataset(ABC):
    """Abstract base class for CJE datasets.

    All dataset implementations should inherit from this class and implement
    the required abstract methods.

    Attributes:
        name: The name of the dataset
    """

    name: str

    @classmethod
    @abstractmethod
    def download(
        cls, cache_dir: Optional[str] = None, split: str = "train"
    ) -> "CJEDataset":
        """Download/prepare dataset and return an instance.

        Args:
            cache_dir: Directory to cache the dataset. If None, uses default cache.
            split: Which split of the dataset to load (e.g. "train", "test").

        Returns:
            An instance of the dataset.
        """
        ...

    @abstractmethod
    def itersamples(self) -> Iterable[CJESample]:
        """Lazily yield CJESample objects.

        Yields:
            CJESample objects representing individual examples.
        """
        ...
