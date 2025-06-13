from typing import List, Tuple, Optional, Union, Dict
import torch
import platform
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging

from cje.loggers.conversation_utils import parse_context, messages_to_text
from cje.utils.logprobs import safe_sum  # NEW

logger = logging.getLogger(__name__)


def _should_use_cpu_for_mps_compatibility() -> bool:
    """
    Check if we should use CPU instead of MPS due to compatibility issues.

    Returns True if we're on macOS < 14.0 and MPS would be used, which has
    issues with torch.isin on Long tensors.
    """
    if not torch.backends.mps.is_available():
        return False

    # Check if we're on macOS and get version
    if platform.system() != "Darwin":
        return False

    try:
        # Get macOS version
        mac_version = platform.mac_ver()[0]
        if mac_version:
            major_version = int(mac_version.split(".")[0])
            # macOS 14.0 (Sonoma) and later should be fine
            if major_version < 14:
                return True
    except (ValueError, IndexError):
        # If we can't parse version, be safe and use CPU
        return True

    return False


class PolicyRunner:
    """
    Wraps a HF causal-LM checkpoint; returns decoded text AND per-token log-probs.
    Now supports conversation parsing and prompt engineering features.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_new_tokens: int = 128,  # Default from generate_with_logp
        temperature: float = 0.0,  # Default from generate_with_logp
        top_p: float = 1.0,  # Default from generate_with_logp
        system_prompt: Optional[str] = None,
        user_message_template: str = "{context}",
        text_format: str = "standard",  # How to format conversations for local models
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Set padding token for GPT-2 style models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Handle device selection with MPS compatibility check
        use_cpu = False
        if device is None:
            if _should_use_cpu_for_mps_compatibility():
                print(
                    "[yellow]Warning: Using CPU instead of MPS due to compatibility issues with torch.isin on macOS < 14.0[/yellow]"
                )
                device_map: Union[str, Dict[str, str]] = {"": "cpu"}
                use_cpu = True
            else:
                device_map = "auto"
        else:
            device_map = {"": device}
            use_cpu = device == "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if use_cpu else torch.float16,
            device_map=device_map,
        )
        # Ensure model's pad token matches tokenizer
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

        # Store generation parameters as attributes
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Store prompt engineering parameters
        self.system_prompt = system_prompt
        self.user_message_template = user_message_template
        self.text_format = text_format

    def _format_context(self, context: str) -> str:
        """Parse context and convert to text format suitable for local models."""
        # Parse the context using shared utilities
        messages = parse_context(
            context, self.system_prompt, self.user_message_template
        )

        # Convert back to text format for local model consumption
        formatted_text = messages_to_text(messages, format_style=self.text_format)

        return formatted_text

    @torch.inference_mode()
    def generate_with_logp(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,  # Allow overriding instance defaults
        temperature: Optional[float] = None,  # Allow overriding instance defaults
        top_p: Optional[float] = None,  # Allow overriding instance defaults
        return_token_logprobs: bool = False,
    ) -> List[Union[Tuple[str, float], Tuple[str, float, List[float]]]]:
        """
        Returns list of ``(decoded_text, sum_logprob)`` for each prompt. When
        ``return_token_logprobs`` is ``True`` each tuple additionally contains a
        list of per-token log probabilities.

        Uses instance defaults for generation params if not provided.
        Now supports conversation parsing and prompt engineering.
        """
        # Use provided params or instance defaults
        gen_max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        )
        gen_temperature = temperature if temperature is not None else self.temperature
        gen_top_p = top_p if top_p is not None else self.top_p

        # Format prompts using conversation parsing and prompt engineering
        formatted_prompts = [self._format_context(prompt) for prompt in prompts]

        inputs = self.tokenizer(
            formatted_prompts, return_tensors="pt", padding=True
        ).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=gen_max_new_tokens,
            temperature=gen_temperature,
            top_p=gen_top_p,
            return_dict_in_generate=True,
            output_scores=True,
        )
        seqs = outputs.sequences  # [B, T]
        scores = outputs.scores  # list[T_new] of [B, V] logits

        results: List[Union[Tuple[str, float], Tuple[str, float, List[float]]]] = []
        for b in range(seqs.size(0)):
            # new tokens start after input length
            start = inputs.input_ids.shape[1]
            toks = seqs[b, start:]
            logp = 0.0
            token_logps: List[float] = []
            for t, s_t in enumerate(scores):
                lp_t = torch.log_softmax(s_t[b], dim=-1)[toks[t]].item()
                logp += lp_t
                if return_token_logprobs:
                    token_logps.append(float(lp_t))
            text = self.tokenizer.decode(seqs[b][start:], skip_special_tokens=True)
            if return_token_logprobs:
                results.append((text, float(logp), token_logps))
            else:
                results.append((text, float(logp)))
        return results

    @torch.inference_mode()
    def log_prob(
        self,
        context: str,
        response: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        *,
        return_token_logprobs: bool = False,
        debug: bool = False,
    ) -> Union[float, Tuple[float, List[float]]]:
        """Return log probability of ``response`` given ``context``.

        The probability is computed under the same decoding settings used by
        :meth:`generate_with_logp`, including temperature and nucleus sampling.
        Now supports conversation parsing and prompt engineering.

        Parameters
        ----------
        context : str
            Prompt/context for the model (may include conversation format).
        response : str
            Response whose probability should be evaluated.
        max_new_tokens : int, optional
            Maximum number of response tokens to consider.  Defaults to the
            instance's ``max_new_tokens`` attribute.
        temperature : float, optional
            Sampling temperature.  Defaults to ``self.temperature``.
        top_p : float, optional
            Nucleus sampling parameter.  Defaults to ``self.top_p``.
        return_token_logprobs : bool, optional
            If True, return a tuple ``(logp, token_logps)`` where ``token_logps`` is the
            list of per-token log-probs (length ≤ ``max_new_tokens``).
        debug : bool, optional
            If True, print debugging information about token probabilities.

        Returns
        -------
        float | Tuple[float, List[float]]
            If ``return_token_logprobs`` is False (default) returns the summed
            log-probability as a single float.  If True returns a tuple
            ``(logp, token_logps)`` where ``token_logps`` is the list of
            per-token log-probs (length ≤ ``max_new_tokens``).
        """

        gen_max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        )
        gen_temperature = temperature if temperature is not None else self.temperature
        gen_top_p = top_p if top_p is not None else self.top_p

        # Format context using conversation parsing and prompt engineering
        formatted_context = self._format_context(context)

        # Important: For consistency with generate_with_logp, we maintain temperature settings
        # but generate() ignores temperature when producing token logits for the actual scores
        # as seen in the warning "temperature generation flag is ignored"
        # So we don't apply the temperature scaling when actually calculating logits

        ctx_ids = self.tokenizer(formatted_context, return_tensors="pt").input_ids.to(
            self.model.device
        )
        resp_ids = self.tokenizer(
            response, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.model.device)

        if debug:
            logger.debug(f"Original context: '{context[:50]}...'")
            logger.debug(f"Formatted context: '{formatted_context[:50]}...'")
            logger.debug(f"Context tokens: {ctx_ids.shape[1]}")
            logger.debug(f"Response tokens: {resp_ids.shape[1]}")
            logger.debug(f"Response text: '{response}'")
            logger.debug(
                f"Total sequence length: {ctx_ids.shape[1] + resp_ids.shape[1]}"
            )

        resp_ids = resp_ids[:, :gen_max_new_tokens]

        from transformers.generation.logits_process import TopPLogitsWarper
        from torch.nn.functional import log_softmax

        warper = TopPLogitsWarper(gen_top_p) if gen_top_p < 1.0 else None

        seq = ctx_ids
        token_logps: List[float] = []

        # Handle empty response case
        if resp_ids.size(1) == 0:
            return 0.0

        for i in range(resp_ids.size(1)):
            logits = self.model(seq).logits[:, -1, :]

            # IMPORTANT: Don't apply temperature scaling - it's not used in the generate() method
            # for scoring according to the warning from generate()

            if warper is not None:
                logits = warper(seq, logits)

            lp = log_softmax(logits, dim=-1)
            token = resp_ids[:, i]

            # Get token log probability, handling potential numerical issues
            token_lp = lp[0, token].item() if torch.isfinite(lp[0, token]) else -100.0

            if debug:
                logger.debug(
                    f"Token {i}: '{self.tokenizer.decode([token.item()])}', logp: {token_lp}"
                )
                if not torch.isfinite(lp[0, token]):
                    logger.warning(
                        f"  Warning: Non-finite logp for token {token.item()}"
                    )
                    top_tokens = torch.topk(logits[0], 5)
                    logger.warning(
                        f"  Top 5 tokens: {[(self.tokenizer.decode([idx.item()]), val.item()) for idx, val in zip(top_tokens.indices, top_tokens.values)]}"
                    )

            token_logps.append(float(token_lp))
            seq = torch.cat([seq, token.unsqueeze(0)], dim=1)

        summed_logp = safe_sum(token_logps)

        if return_token_logprobs:
            return summed_logp, token_logps
        else:
            return summed_logp
