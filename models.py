import os
import csv
import sys
import torch
import torch.nn.functional as F  # Added for Entropy math
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

# --- [THESIS UTILITIES] ---

# Log file for thesis probe (set THESIS_PROBE_LOG to override path)
_THESIS_PROBE_LOG_PATH: Optional[str] = None

# Optional stats collector for --compare_quantizations (set by run.py, read after each run)
_quantization_run_stats: Optional[dict] = None


def set_quantization_stats_collector(stats: Optional[dict]) -> None:
    """Enable or disable collection of bandwidth stats for comparison table."""
    global _quantization_run_stats
    _quantization_run_stats = stats


def get_and_reset_quantization_stats() -> Tuple[float, int]:
    """Return (total_transmitted_mb, num_steps) and reset the collector."""
    global _quantization_run_stats
    if _quantization_run_stats is None:
        return 0.0, 0
    total = _quantization_run_stats.get("total_transmitted_mb", 0.0)
    steps = _quantization_run_stats.get("num_steps", 0)
    _quantization_run_stats = None
    return total, steps


def _get_thesis_probe_log_path() -> str:
    global _THESIS_PROBE_LOG_PATH
    if _THESIS_PROBE_LOG_PATH is not None:
        return _THESIS_PROBE_LOG_PATH
    path = os.environ.get("THESIS_PROBE_LOG")
    if path:
        _THESIS_PROBE_LOG_PATH = path
        return path
    # Default: logs/thesis_probe.log under current working directory
    log_dir = os.path.join(os.getcwd(), "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "thesis_probe.log")
    except OSError:
        path = os.path.join(os.getcwd(), "thesis_probe.log")
    _THESIS_PROBE_LOG_PATH = path
    return path


def probe_latent_overhead(
    hidden_states: torch.Tensor, step_label: str, quant_bits: int = 16
):
    """
    Measures Bandwidth AND Entropy of the latent tensor.
    This provides both 'cost' and 'information density' data for thesis tables.
    Logs to stdout and appends to a file (logs/thesis_probe.log by default).

    Args:
        hidden_states: The latent tensor to measure
        step_label: Label for this probe point
        quant_bits: Quantization bits (2, 4, 8, or 16) - used to calculate simulated transmitted size
    """
    num_elements = hidden_states.nelement()
    element_size = (
        hidden_states.element_size()
    )  # Native size (usually 2 bytes for bfloat16)
    raw_size_mb = (num_elements * element_size) / (1024 * 1024)

    # Calculate simulated transmitted size based on quantization
    # Native is 16-bit (bfloat16), so scale by quant_bits/16
    transmitted_size_mb = raw_size_mb * (quant_bits / 16.0)

    # Calculate Entropy to prove Information Density changes
    current_entropy = calculate_latent_entropy(hidden_states)

    shape_str = str(list(hidden_states.shape))
    msg = (
        f">>> [THESIS PROBE] {step_label} | "
        f"Entropy: {current_entropy:.4f} | "
        f"Raw: {raw_size_mb:.4f} MB | "
        f"Transmitted ({quant_bits}-bit): {transmitted_size_mb:.4f} MB | "
        f"Shape: {shape_str}"
    )
    print(msg, flush=True)
    try:
        log_path = _get_thesis_probe_log_path()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} {msg}\n")
    except OSError:
        pass
    # Accumulate for --compare_quantizations table
    if _quantization_run_stats is not None:
        _quantization_run_stats["total_transmitted_mb"] = (
            _quantization_run_stats.get("total_transmitted_mb", 0.0) + transmitted_size_mb
        )
        _quantization_run_stats["num_steps"] = (
            _quantization_run_stats.get("num_steps", 0) + 1
        )
    return raw_size_mb, transmitted_size_mb


# --- [NOVELTY UTILITIES] ---


def calculate_latent_entropy(hidden_states: torch.Tensor, lm_head=None) -> float:
    """
    Calculates the Shannon Entropy of the model's prediction distribution.
    
    HIGH ENTROPY = HIGH COMPLEXITY (model is uncertain, exploring many options)
    LOW ENTROPY = LOW COMPLEXITY (model is confident, clear answer)
    
    Args:
        hidden_states: The latent tensor [batch, hidden_dim] or [batch, seq, hidden_dim]
        lm_head: Optional LM head to project to vocabulary space.
                 If None, uses hidden state softmax (less meaningful but still works).
        
    Returns:
        Mean entropy value in bits (log base 2) across all positions
    """
    with torch.no_grad():
        if lm_head is not None:
            # PROJECT to vocabulary space for TRUE prediction entropy
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)
            
            logits = lm_head(hidden_states)  # [batch, seq, vocab_size]
            last_logits = logits[:, -1, :]  # [batch, vocab_size]
            probs = F.softmax(last_logits.float(), dim=-1)
        else:
            # Fallback: use hidden state distribution (less meaningful)
            probs = F.softmax(hidden_states.float(), dim=-1)
        
        # Shannon Entropy in BITS: -sum(p * log2(p))
        log_probs = torch.log2(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        return entropy.mean().item()


def latent_sieve_quantize(hidden_states: torch.Tensor, bits: int = 16):
    """
    The 'Neural Sieve': Simulates low-bandwidth communication by quantizing tensors.
    Bits = 16: No change (Baseline)
    Bits = 8: 50% bandwidth reduction
    Bits = 4: 75% bandwidth reduction
    Bits = 2: 87.5% bandwidth reduction
    """
    if bits >= 16:
        return hidden_states

    # Calculate scale and min/max for uniform quantization
    min_val = hidden_states.min()
    max_val = hidden_states.max()
    q_range = (2**bits) - 1
    scale = (max_val - min_val) / q_range

    # Quantize to lower precision and then simulate the 'received' state
    quantized = torch.round((hidden_states - min_val) / scale).to(
        torch.uint8 if bits <= 8 else torch.int32
    )
    dequantized = (quantized.to(hidden_states.dtype) * scale) + min_val

    return dequantized


# --- [ORIGINAL HELPER FUNCTIONS] ---


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values) -> int:
    """Get the sequence length from past_key_values, handling both tuple and DynamicCache formats."""
    if past_key_values is None:
        return 0
    # Handle DynamicCache (transformers >= 4.36)
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    # Handle legacy tuple format
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        first_layer = past_key_values[0]
        if isinstance(first_layer, (tuple, list)) and len(first_layer) > 0:
            k = first_layer[0]
            return k.shape[-2]
    return 0


# --- [MAIN MODEL WRAPPER] ---


class ModelWrapper:
    def __init__(
        self, model_name: str, device: torch.device, use_vllm: bool = False, args=None
    ):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = (
            bool(getattr(args, "latent_space_realign", False)) if args else False
        )
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args

        # Thesis Configuration
        self.quant_bits = int(getattr(args, "quant_bits", 16))

        if self.use_vllm:
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))

            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas":
                self.vllm_engine = LLM(
                    model=model_name,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=gpu_util,
                    enable_prefix_caching=True,
                    enable_prompt_embeds=True,
                )
            else:
                self.vllm_engine = LLM(
                    model=model_name,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=gpu_util,
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

            use_second_hf = (
                bool(getattr(args, "use_second_HF_model", False)) if args else False
            )
            if use_second_hf:
                self.HF_model = (
                    AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=(
                            torch.bfloat16
                            if torch.cuda.is_available()
                            else torch.float32
                        ),
                    )
                    .to(args.device2)
                    .eval()
                )
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                self._ensure_latent_realign_matrix(
                    self.HF_model, torch.device(self.HF_device), args
                )
            _ensure_pad_token(self.tokenizer)
            return

        # Normal Transformers Path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
            )
        self.model.to(device).eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

    def _build_latent_realign_matrix(self, model, device, args):
        in_emb = (
            model.get_input_embeddings().weight.detach().to(device, dtype=torch.float32)
        )
        out_emb = (
            model.get_output_embeddings()
            .weight.detach()
            .to(device, dtype=torch.float32)
        )
        gram = torch.matmul(out_emb.T, out_emb) + 1e-5 * torch.eye(
            out_emb.shape[1], device=device
        )
        rhs = torch.matmul(out_emb.T, in_emb)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = in_emb.norm(dim=1).mean()

        if not self.latent_space_realign:
            realign_matrix = torch.eye(realign_matrix.shape[0], device=device)
        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(
        self, model, device, args
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(
                model, target_device, args
            )
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = (
            target_norm.to(device=target_device, dtype=matrix.dtype)
            if isinstance(target_norm, torch.Tensor)
            else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        )
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(
        self, hidden: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(
            model, hidden.device, self.args
        )
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )[
            "input_ids"
        ].to(self.device)

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        """
        Convert a batch of chat message lists into prompts, input_ids, attention_mask,
        and tokens_batch for downstream forward/generation.

        Args:
            batch_messages: List of message lists; each is [{"role": "...", "content": "..."}, ...].
            add_generation_prompt: Whether to add the assistant turn prompt.

        Returns:
            prompts: List of prompt strings.
            input_ids: [batch, seq_len] tensor on self.device.
            attention_mask: [batch, seq_len] tensor on self.device.
            tokens_batch: List of list of token strings per item (after masking padding).
        """
        prompts = []
        for messages in batch_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            if isinstance(text, list):
                text = text[0] if text else ""
            prompts.append(text)
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active))
        return prompts, input_ids, attention_mask, tokens_batch

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values

        e_t = outputs.hidden_states[0][:, -1, :]  # [B, D]
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]
        h_t = last_hidden.detach().clone()

        # [THESIS PROBE] Record baseline size and entropy (used by latent_mas_hybrid path)
        probe_latent_overhead(last_hidden, "Latent Entry", self.quant_bits)

        e_t_plus_1 = None
        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())

        for step in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            
            # 1. Apply Latent Realignment
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)

            # 2. [THESIS SIEVE] Apply Quantization
            # Future Novelty: If entropy > threshold, use self.quant_bits + 2
            latent_vec = latent_sieve_quantize(latent_vec, bits=self.quant_bits)

            latent_vecs_all.append(latent_vec.detach().clone())

            if step == 0:
                e_t_plus_1 = latent_vec.detach().clone()

            # 3. Roll out the next step
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            
            # [THESIS PROBE] Monitor complexity drift during reasoning
            probe_latent_overhead(last_hidden, f"Latent Step {step}", self.quant_bits)

        return past

    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        curr_output_embedding = []
        curr_output_embedding.append(outputs.hidden_states[0])  # input embedding

        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1)  # Output input embeddings
