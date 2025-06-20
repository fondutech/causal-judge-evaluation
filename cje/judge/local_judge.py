from __future__ import annotations
import logging
from typing import List, Dict, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from .base import LocalJudgeConfig
from .judges import Judge
from .schemas import JudgeScore

logger = logging.getLogger(__name__)


class LocalJudge(Judge):
    """Local model-based judge supporting configurable models."""

    def __init__(self, config: LocalJudgeConfig):
        self.config: LocalJudgeConfig = config
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._device = self._setup_device()

    def _setup_device(self) -> torch.device:
        """Setup the appropriate device for the model."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)

    def _get_torch_dtype(self) -> torch.dtype:
        """Get the appropriate torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.config.torch_dtype, torch.float16)

    def _lazy_load(self) -> None:
        """Lazy load the model and tokenizer."""
        if self._model is None:
            logger.info(f"Loading model {self.config.model_name} on {self._device}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, use_fast=True
            )

            torch_dtype = self._get_torch_dtype()

            if self._device.type == "cuda":
                # Use device_map for multi-GPU setups
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # Single device setup
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )
                self._model.to(self._device)

            self._model.eval()
            logger.info("Model loaded successfully")

    def _prepare_input(self, context: str, response: str) -> str:
        """Prepare input for the specific model."""
        # Different models may need different input formats
        model_name_lower = self.config.model_name.lower()

        if "prometheus" in model_name_lower:
            # Prometheus-specific format
            import json

            return f"<|system|>\n{json.dumps({'instruction': context})}\n<|assistant|>\n{response}"
        else:
            # Generic format - use template
            # For now, use simple concatenation
            return f"Context: {context}\nResponse: {response}"

    def score(self, context: str, response: str) -> JudgeScore:
        """Score a single context-response pair."""
        self._lazy_load()
        assert self._tokenizer is not None
        assert self._model is not None

        text = self._prepare_input(context, response)

        # Tokenize
        max_length = getattr(self._tokenizer, "model_max_length", 2048) or 2048
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self._device)

        # Get score
        with torch.inference_mode():
            outputs = self._model(**inputs)

            # Handle different output formats
            if hasattr(outputs, "logits"):
                # Classification head output
                logits = outputs.logits
                if logits.dim() == 2 and logits.size(1) == 1:
                    # Single regression output
                    score = logits.squeeze().item()
                else:
                    # Classification outputs - take mean or max
                    score = torch.softmax(logits, dim=-1).max().item()
            else:
                # Direct scalar output
                score = outputs.item()

        # Convert to JudgeScore (deterministic for local models)
        return JudgeScore(mean=float(score), variance=0.0)

    def score_batch(
        self, samples: List[Dict[str, str]], disable_progress: bool = False
    ) -> List[JudgeScore]:
        """Score a batch of context-response pairs efficiently."""
        self._lazy_load()
        assert self._tokenizer is not None
        assert self._model is not None

        scores = []

        # Process in batches
        for i in tqdm(
            range(0, len(samples), self.config.batch_size),
            desc=f"Scoring with {self.config.model_name}",
            disable=disable_progress or len(samples) < 10,
        ):
            batch = samples[i : i + self.config.batch_size]
            texts = [
                self._prepare_input(sample["context"], sample["response"])
                for sample in batch
            ]

            # Tokenize batch
            max_length = getattr(self._tokenizer, "model_max_length", 2048) or 2048
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self._device)

            # Get scores
            with torch.inference_mode():
                outputs = self._model(**inputs)

                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                    if logits.dim() == 2 and logits.size(1) == 1:
                        # Regression outputs
                        batch_scores = logits.squeeze(-1).cpu().tolist()
                    else:
                        # Classification outputs
                        probs = torch.softmax(logits, dim=-1)
                        batch_scores = probs.max(dim=-1)[0].cpu().tolist()
                else:
                    # Direct outputs
                    batch_scores = outputs.cpu().tolist()

            scores.extend(batch_scores)

        # Convert to JudgeScore objects
        return [JudgeScore(mean=float(score), variance=0.0) for score in scores]
