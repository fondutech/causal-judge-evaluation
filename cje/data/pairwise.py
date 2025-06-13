"""Pairwise comparison dataset adapter for CJE with Bradley-Terry model."""

from pathlib import Path
from typing import Optional, Iterator, Dict, List, Tuple
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.special import expit

from .base import CJEDataset
from .schema import CJESample


_CACHE = Path.home() / ".cache" / "cje" / "pairwise"
_CACHE.mkdir(parents=True, exist_ok=True)


class BradleyTerryModel:
    """Bradley-Terry model for converting pairwise comparisons to utility scores.

    The Bradley-Terry model assumes that the probability of item i beating item j is:
    P(i > j) = exp(u_i) / (exp(u_i) + exp(u_j))

    where u_i and u_j are the latent utility scores.
    """

    def __init__(self, regularization: float = 0.01):
        """Initialize the Bradley-Terry model.

        Args:
            regularization: L2 regularization strength for utility scores
        """
        self.regularization = regularization
        self.utilities: Optional[np.ndarray] = None
        self.item_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_item: Optional[Dict[int, str]] = None

    def fit(self, comparisons: List[Tuple[str, str, float]]) -> None:
        """Fit the Bradley-Terry model to pairwise comparison data.

        Args:
            comparisons: List of (winner_id, loser_id, weight) tuples.
                        weight can be used for ties (0.5) or confidence
        """
        # Build item index mapping
        unique_items = set()
        for winner, loser, _ in comparisons:
            unique_items.add(winner)
            unique_items.add(loser)

        self.item_to_idx = {item: idx for idx, item in enumerate(sorted(unique_items))}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        n_items = len(unique_items)

        # Convert comparisons to indices
        indexed_comparisons = []
        for winner, loser, weight in comparisons:
            indexed_comparisons.append(
                (self.item_to_idx[winner], self.item_to_idx[loser], weight)
            )

        # Define negative log-likelihood
        def neg_log_likelihood(utilities: np.ndarray) -> float:
            ll = 0
            for i, j, w in indexed_comparisons:
                # P(i beats j) = exp(u_i) / (exp(u_i) + exp(u_j))
                # Log likelihood: w * log(P(i>j)) + (1-w) * log(P(j>i))
                log_p_i_wins = utilities[i] - np.logaddexp(utilities[i], utilities[j])
                log_p_j_wins = utilities[j] - np.logaddexp(utilities[i], utilities[j])
                ll += w * log_p_i_wins + (1 - w) * log_p_j_wins

            # Add L2 regularization (anchor to zero mean)
            ll -= self.regularization * np.sum(utilities**2)

            return -ll

        # Gradient of negative log-likelihood
        def gradient(utilities: np.ndarray) -> np.ndarray:
            grad = np.zeros(n_items)

            for i, j, w in indexed_comparisons:
                p_i_wins = expit(utilities[i] - utilities[j])
                grad[i] += w - p_i_wins
                grad[j] += (1 - w) - (1 - p_i_wins)

            # Regularization gradient
            grad -= 2 * self.regularization * utilities

            return -grad

        # Initialize utilities to zero
        initial_utilities = np.zeros(n_items)

        # Optimize
        result = minimize(
            neg_log_likelihood,
            initial_utilities,
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": 1000},
        )

        if not result.success:
            warnings.warn(
                f"Bradley-Terry optimization did not converge: {result.message}"
            )

        self.utilities = result.x

        # Normalize utilities to [0, 1] range for compatibility with CJE
        # Using sigmoid transformation to maintain relative ordering
        self.utilities = expit(self.utilities)

    def get_utility(self, item_id: str) -> float:
        """Get the utility score for an item.

        Args:
            item_id: ID of the item

        Returns:
            Utility score in [0, 1]
        """
        if self.utilities is None:
            raise ValueError("Model must be fitted before getting utilities")

        if self.item_to_idx is None or item_id not in self.item_to_idx:
            # Return neutral utility for unseen items
            return 0.5

        return float(self.utilities[self.item_to_idx[item_id]])


class PairwiseComparisonDataset(CJEDataset):
    """Adapter for pairwise comparison datasets like chatbot_arena_conversations.

    This dataset contains pairwise human preferences which are converted to
    scalar utilities using the Bradley-Terry model.
    """

    name = "PairwiseComparison"

    def __init__(self, tbl: pa.Table, bt_model: BradleyTerryModel) -> None:
        """Initialize the dataset.

        Args:
            tbl: PyArrow table containing the dataset
            bt_model: Fitted Bradley-Terry model for utility conversion
        """
        self._tbl = tbl
        self._bt_model = bt_model

    @classmethod
    def download(
        cls,
        cache_dir: Optional[str] = None,
        split: Optional[str] = None,
        dataset_name: str = "agie-ai/lmsys-chatbot_arena_conversations",
        regularization: float = 0.01,
    ) -> "PairwiseComparisonDataset":
        """Download and process a pairwise comparison dataset.

        Args:
            cache_dir: Directory to cache the dataset. If None, uses default cache.
            split: Dataset split to load. If None, loads all available data.
            dataset_name: Name of the HuggingFace dataset to load
            regularization: L2 regularization for Bradley-Terry model

        Returns:
            A PairwiseComparisonDataset instance.
        """
        cache_path = _CACHE if cache_dir is None else Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Load dataset
        parquet_path = (
            cache_path / f"{dataset_name.replace('/', '_')}_{split or 'all'}.parquet"
        )

        if not parquet_path.exists():
            if split is None:
                # Load full dataset
                ds = load_dataset(dataset_name)
                # Concatenate all splits
                from datasets import concatenate_datasets

                all_datasets = []
                for split_name, dataset in ds.items():
                    dataset_with_split = dataset.add_column(
                        "split", [split_name] * len(dataset)
                    )
                    all_datasets.append(dataset_with_split)
                full_dataset = concatenate_datasets(all_datasets)
                full_dataset.to_parquet(str(parquet_path))
            else:
                # Load specific split
                ds = load_dataset(dataset_name, split=split)
                ds_with_split = ds.add_column("split", [split] * len(ds))
                ds_with_split.to_parquet(str(parquet_path))

        tbl = pq.read_table(str(parquet_path))

        # Extract pairwise comparisons and fit Bradley-Terry model
        comparisons = cls._extract_comparisons(tbl)
        bt_model = BradleyTerryModel(regularization=regularization)
        bt_model.fit(comparisons)

        return cls(tbl, bt_model)

    @staticmethod
    def _extract_comparisons(tbl: pa.Table) -> List[Tuple[str, str, float]]:
        """Extract pairwise comparisons from the dataset.

        This method should be adapted based on the specific dataset schema.
        For chatbot_arena_conversations, we expect fields like:
        - conversation_a, conversation_b: The two responses being compared
        - winner: Which response won (could be "A", "B", or "tie")

        Returns:
            List of (winner_id, loser_id, weight) tuples
        """
        comparisons = []

        for row in tqdm(tbl.to_pylist(), desc="Extracting comparisons"):
            # Generate unique IDs for each response
            # This assumes the dataset has some way to identify models/responses
            conv_id = row.get("conversation_id", row.get("id", None))

            if conv_id is None:
                continue

            # Create unique IDs for the two options
            id_a = f"{conv_id}_model_a"
            id_b = f"{conv_id}_model_b"

            winner = row.get("winner", row.get("preference", None))

            if winner == "model_a" or winner == "A":
                comparisons.append((id_a, id_b, 1.0))
            elif winner == "model_b" or winner == "B":
                comparisons.append((id_b, id_a, 1.0))
            elif winner == "tie":
                # For ties, we add both directions with weight 0.5
                comparisons.append((id_a, id_b, 0.5))
                comparisons.append((id_b, id_a, 0.5))

        return comparisons

    def itersamples(self, disable_progress: bool = False) -> Iterator[CJESample]:
        """Yield CJESamples with Bradley-Terry utilities as y_true.

        Each conversation becomes two CJESamples (one for each model response).

        Args:
            disable_progress: If True, disable the progress bar.
        """
        for row in tqdm(self._tbl.to_pylist(), disable=disable_progress):
            conv_id = row.get("conversation_id", row.get("id", None))
            if conv_id is None:
                continue

            # Extract context (the prompt/question)
            context = row.get("prompt", row.get("question", ""))

            # Extract the two responses
            response_a = row.get("response_a", row.get("conversation_a", []))
            response_b = row.get("response_b", row.get("conversation_b", []))

            # Convert to string if needed
            if isinstance(response_a, list):
                response_a = "\n".join(
                    [
                        msg.get("content", "")
                        for msg in response_a
                        if isinstance(msg, dict)
                    ]
                )
            if isinstance(response_b, list):
                response_b = "\n".join(
                    [
                        msg.get("content", "")
                        for msg in response_b
                        if isinstance(msg, dict)
                    ]
                )

            # Get Bradley-Terry utilities
            id_a = f"{conv_id}_model_a"
            id_b = f"{conv_id}_model_b"

            utility_a = self._bt_model.get_utility(id_a)
            utility_b = self._bt_model.get_utility(id_b)

            # Yield sample for model A
            yield CJESample(
                uid=id_a,
                context=context,
                response=response_a,
                y_true=utility_a,
                logp=None,  # Would need to be added based on model logs
                meta={
                    "conversation_id": conv_id,
                    "model_label": "model_a",
                    "winner": row.get("winner", row.get("preference", None)),
                    "split": row.get("split", "unknown"),
                    "bt_utility": utility_a,
                },
            )

            # Yield sample for model B
            yield CJESample(
                uid=id_b,
                context=context,
                response=response_b,
                y_true=utility_b,
                logp=None,
                meta={
                    "conversation_id": conv_id,
                    "model_label": "model_b",
                    "winner": row.get("winner", row.get("preference", None)),
                    "split": row.get("split", "unknown"),
                    "bt_utility": utility_b,
                },
            )

    def __len__(self) -> int:
        """Return the number of CJESamples (2x number of conversations)."""
        # Each conversation yields 2 samples
        return len(self._tbl) * 2

    def get_bt_model(self) -> BradleyTerryModel:
        """Get the fitted Bradley-Terry model for further analysis."""
        return self._bt_model
