"""
Core CJE functionality in one place - simpler imports and clearer structure.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass, asdict

from .config.simple import CJEConfig, PolicyConfig
from .providers.unified import UnifiedProvider
from .estimators.base_crossfit import BaseCrossFittedEstimator
from .data.base import CJEDataset
from .utils.progress import console


@dataclass
class CJEResult:
    """Simplified result object."""

    estimates: np.ndarray  # Shape: (n_policies,)
    std_errors: np.ndarray  # Shape: (n_policies,)
    policy_names: List[str]
    sample_size: int
    estimator_type: str
    ess_percentage: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def summary(self) -> str:
        """Get a text summary of results."""
        lines = [f"CJE Results ({self.estimator_type}, n={self.sample_size})"]
        lines.append("-" * 50)

        for i, name in enumerate(self.policy_names):
            est = self.estimates[i]
            se = self.std_errors[i]
            ci_lower = est - 1.96 * se
            ci_upper = est + 1.96 * se

            ess_str = ""
            if self.ess_percentage is not None:
                ess = self.ess_percentage[i]
                ess_str = f" (ESS: {ess:.1f}%)"

            lines.append(
                f"{name}: {est:.4f} ± {se:.4f} "
                f"[{ci_lower:.4f}, {ci_upper:.4f}]{ess_str}"
            )

        return "\n".join(lines)

    def to_json(self, path: Path) -> None:
        """Save results to JSON."""
        data = {
            "estimates": self.estimates.tolist(),
            "std_errors": self.std_errors.tolist(),
            "policy_names": self.policy_names,
            "sample_size": self.sample_size,
            "estimator_type": self.estimator_type,
            "ess_percentage": (
                self.ess_percentage.tolist()
                if self.ess_percentage is not None
                else None
            ),
            "metadata": self.metadata,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "CJEResult":
        """Load results from JSON."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            estimates=np.array(data["estimates"]),
            std_errors=np.array(data["std_errors"]),
            policy_names=data["policy_names"],
            sample_size=data["sample_size"],
            estimator_type=data["estimator_type"],
            ess_percentage=(
                np.array(data["ess_percentage"]) if data.get("ess_percentage") else None
            ),
            metadata=data.get("metadata", {}),
        )


class CJEPipeline:
    """
    Simplified main pipeline for CJE evaluation.

    This replaces the complex pipeline.py with a cleaner interface.
    """

    def __init__(self, config: CJEConfig):
        self.config = config
        self.work_dir = Path(config.work_dir)
        self.work_dir.mkdir(exist_ok=True)

        # Initialize providers
        self.logging_provider = UnifiedProvider(
            config.logging_policy.provider, config.logging_policy.model_name
        )

        self.target_providers = [
            UnifiedProvider(p.provider, p.model_name) for p in config.target_policies
        ]

        if config.judge:
            self.judge_provider = UnifiedProvider(
                config.judge.provider, config.judge.model_name
            )
        else:
            self.judge_provider = None

    def run(self) -> CJEResult:
        """Run the complete CJE pipeline."""
        console.print("[bold blue]Starting CJE Pipeline[/bold blue]")

        # 1. Load data
        console.print("Loading dataset...")
        dataset = self._load_dataset()

        # 2. Compute/load log probabilities
        console.print("Computing importance weights...")
        weights = self._compute_weights(dataset)

        # 3. Get judge scores
        console.print("Getting judge scores...")
        rewards = self._get_rewards(dataset)

        # 4. Run estimation
        console.print("Running causal estimation...")
        result = self._run_estimation(dataset, weights, rewards)

        # 5. Save results
        output_path = self.work_dir / "results.json"
        result.to_json(output_path)
        console.print(f"[green]Results saved to {output_path}[/green]")

        # Print summary
        console.print("\n" + result.summary())

        return result

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset based on config."""
        # Simplified dataset loading
        if self.config.dataset.name == "ChatbotArena":
            from .data.chatbot_arena import ChatbotArenaDataset

            dataset_obj = ChatbotArenaDataset.download(split=self.config.dataset.split)
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset.name}")

        # Get samples from dataset
        all_samples = []
        for sample in dataset_obj.itersamples():
            all_samples.append(sample)

        # Limit samples if specified
        if self.config.dataset.sample_limit:
            all_samples = all_samples[: self.config.dataset.sample_limit]

        # Convert to dict format
        data = []
        for sample in all_samples:
            data.append(
                {
                    "context": sample.context,
                    "response": sample.response,
                    "y_true": sample.y_true,
                    "logp": sample.logp,
                    "meta": sample.meta,
                    "uid": sample.uid,
                }
            )

        return data

    def _compute_weights(self, dataset: List[Dict[str, Any]]) -> np.ndarray:
        """Compute importance weights matrix."""
        n = len(dataset)
        k = len(self.config.target_policies)
        weights = np.zeros((n, k))

        for i, item in enumerate(dataset):
            context = item["context"]
            response = item["response"]

            # Get logging policy probability
            log_p0 = self.logging_provider.score(context, response)

            # Get target policy probabilities
            for j, target_provider in enumerate(self.target_providers):
                log_p1 = target_provider.score(context, response)
                weights[i, j] = np.exp(log_p1 - log_p0)

        # Apply clipping
        if self.config.estimator.clip:
            weights = np.minimum(weights, self.config.estimator.clip)

        return weights

    def _get_rewards(self, dataset: List[Dict[str, Any]]) -> np.ndarray:
        """Get judge scores for the dataset."""
        rewards = np.zeros(len(dataset))

        for i, item in enumerate(dataset):
            if "reward" in item:
                # Use existing reward
                rewards[i] = item["reward"]
            elif self.judge_provider:
                # Score with judge
                score = self.judge_provider.judge(
                    item["context"],
                    item["response"],
                    template=self.config.judge.template,
                )
                rewards[i] = score
            else:
                # Default reward
                rewards[i] = 0.5

        return rewards

    def _run_estimation(
        self, dataset: List[Dict[str, Any]], weights: np.ndarray, rewards: np.ndarray
    ) -> CJEResult:
        """Run the selected estimator."""
        # Create simplified logs format
        logs = []
        for i, item in enumerate(dataset):
            logs.append(
                {
                    "context": item["context"],
                    "response": item["response"],
                    "reward": rewards[i],
                    "logp": 0.0,  # Placeholder, not used in simplified version
                }
            )

        # Choose estimator
        if self.config.estimator.name == "DRCPO":
            from .estimators.drcpo import MultiDRCPOEstimator

            # Create mock sampler that returns pre-computed weights
            sampler = SimpleSampler(weights, self.config.target_policies)
            estimator = MultiDRCPOEstimator(
                sampler=sampler,  # type: ignore[arg-type]
                k=self.config.estimator.k_folds,
                clip=self.config.estimator.clip,
                calibrate_weights=self.config.estimator.calibrate_weights,
                calibrate_outcome=self.config.estimator.calibrate_outcome,
            )
        else:
            raise ValueError(f"Unknown estimator: {self.config.estimator.name}")

        # Fit and estimate
        estimator.fit(logs)
        est_result = estimator.estimate()

        # Convert to simplified result
        return CJEResult(
            estimates=est_result.v_hat,
            std_errors=est_result.se,
            policy_names=[p.name for p in self.config.target_policies],
            sample_size=len(dataset),
            estimator_type=self.config.estimator.name,
            ess_percentage=est_result.metadata.get("ess_percentage"),
            metadata=est_result.metadata,
        )


class SimpleSampler:
    """Simple sampler that returns pre-computed weights."""

    def __init__(self, weights: np.ndarray, policies: List[PolicyConfig]):
        self.weights = weights
        self.K = len(policies)  # Required by MultiDRCPOEstimator
        self.policy_names = [p.name for p in policies]
        self._target_policies = policies  # Store for compatibility

    def importance_weights_matrix(
        self,
        contexts: List[str],
        responses: List[str],
        logp_behavior: List[float],
        clip: Optional[float] = None,
        stabilize: bool = True,
        return_stats: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return pre-computed weights with stats."""
        stats = {
            "ess_values": [],
            "ess_percentage": [],
            "n_clipped": 0,
            "clip_fraction": 0.0,
            "weight_range": [np.min(self.weights), np.max(self.weights)],
            "stabilization_applied": False,
        }

        # Compute ESS for each policy
        ess_values: List[float] = []
        ess_percentage: List[float] = []
        for k in range(self.K):
            w_k = self.weights[:, k]
            ess = np.sum(w_k) ** 2 / np.sum(w_k**2)
            ess_values.append(float(ess))
            ess_percentage.append(float(100 * ess / len(w_k)))

        stats["ess_values"] = ess_values
        stats["ess_percentage"] = ess_percentage

        return self.weights, stats

    def sample_many(self, context: str, n: int = 1) -> List[List[str]]:
        """Mock sampling - just return empty responses."""
        # Return n empty responses for each policy
        return [["" for _ in range(n)] for _ in range(self.K)]


def run_cje(config_path: str) -> CJEResult:
    """Main entry point - run CJE from config file."""
    config = CJEConfig.from_yaml(Path(config_path))
    pipeline = CJEPipeline(config)
    return pipeline.run()
