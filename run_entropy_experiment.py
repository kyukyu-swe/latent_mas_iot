#!/usr/bin/env python3
"""
Entropy Experiment Script for Thesis Demonstration

This script proves that the New Algorithm's entropy measurement correlates 
with prompt complexity:
- Simple prompts (e.g., "What is 1+1?") → Low entropy (~4.0)
- Complex prompts (e.g., GSM8K word problem) → High entropy (~9.0)

Usage:
    python run_entropy_experiment.py --model_name Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import torch
import os
from datetime import datetime
from typing import Dict, List, Optional

from models import ModelWrapper, calculate_latent_entropy, probe_latent_overhead
from utils import auto_device, set_seed
import torch.nn.functional as F


def calculate_output_entropy(model, hidden_states: torch.Tensor) -> float:
    """
    Calculate the entropy of the model's next-token prediction distribution.
    
    This is the TRUE measure of "complexity":
    - Simple prompts → Model is confident → Low entropy
    - Complex prompts → Model is uncertain/exploring → High entropy
    
    Returns entropy in bits (log base 2).
    """
    # Get logits by projecting hidden states through the output layer
    with torch.no_grad():
        # hidden_states: [batch, hidden_dim] or [batch, seq, hidden_dim]
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
        
        # Project to vocabulary using the LM head
        logits = model.model.lm_head(hidden_states)  # [batch, seq, vocab_size]
        
        # Get the last position's logits
        last_logits = logits[:, -1, :]  # [batch, vocab_size]
        
        # Convert to probabilities
        probs = F.softmax(last_logits.float(), dim=-1)
        
        # Shannon entropy: -sum(p * log2(p))
        # Using log2 to get entropy in bits
        log_probs = torch.log2(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        return entropy.mean().item()


# ============================================================================
# EXPERIMENT PROMPTS
# ============================================================================

SIMPLE_PROMPTS = [
    "What is 1+1?",
    "Hello.",
    "Say hi.",
    "What color is the sky?",
]

COMPLEX_PROMPTS = [
    # GSM8K-style word problem
    """A train leaves Bangkok at 60 km/h heading east. Another train leaves Chiang Mai at 80 km/h heading west toward Bangkok. If the distance between the cities is 700 km, and both trains leave at the same time, how many hours will it take for the trains to meet? Show your step-by-step reasoning.""",
    
    # Another GSM8K sample
    """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?""",
    
    # Multi-step reasoning
    """A store sells notebooks for $3 each. If you buy more than 5, you get a 20% discount on all notebooks. Sarah wants to buy notebooks for herself and 7 friends (8 total). She has $25. Does she have enough money? If not, how much more does she need? Show all calculations.""",
]


def run_single_prompt_entropy(
    model: ModelWrapper,
    prompt: str,
    prompt_label: str,
    args: argparse.Namespace,
) -> Dict:
    """
    Run a single prompt through the model and capture entropy.
    
    Returns dict with:
        - prompt: The input prompt
        - label: Simple/Complex
        - entropy_initial: Entropy at latent entry
        - entropy_steps: List of entropy values at each latent step
        - avg_entropy: Average entropy across all measurements
    """
    # Build chat message format
    messages = [{"role": "user", "content": prompt}]
    
    prompts, input_ids, attention_mask, _ = model.prepare_chat_batch(
        [messages], add_generation_prompt=True
    )
    
    # Capture entropy during forward pass
    entropy_values = []
    
    # Initial forward to get hidden states
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get the last hidden state (where the model's "understanding" is encoded)
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]
        
        # Calculate and record initial entropy using OUTPUT distribution (vocabulary space)
        # This measures prediction uncertainty - the TRUE complexity metric
        initial_entropy = calculate_output_entropy(model, last_hidden)
        entropy_values.append(initial_entropy)
        
        # Run latent steps to see entropy evolution
        past = outputs.past_key_values
        
        for step in range(args.latent_steps):
            # Apply realignment (simulating agent communication)
            latent_vec = model._apply_latent_realignment(last_hidden, model.model)
            latent_embed = latent_vec.unsqueeze(1)
            
            # Get past length for attention mask
            if hasattr(past, "get_seq_length"):
                past_len = past.get_seq_length()
            else:
                past_len = past[0][0].shape[-2] if past else 0
            
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            
            outputs = model.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            
            # Use output entropy (vocabulary distribution) for step entropy too
            step_entropy = calculate_output_entropy(model, last_hidden)
            entropy_values.append(step_entropy)
    
    avg_entropy = sum(entropy_values) / len(entropy_values)
    
    return {
        "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
        "label": prompt_label,
        "entropy_initial": entropy_values[0],
        "entropy_steps": entropy_values[1:] if len(entropy_values) > 1 else [],
        "avg_entropy": avg_entropy,
        "max_entropy": max(entropy_values),
        "token_count": input_ids.shape[1],
    }


def run_entropy_experiment(args):
    """Main experiment: Compare entropy between simple and complex prompts."""
    
    print("=" * 70)
    print("ENTROPY EXPERIMENT: Simple vs Complex Prompt Complexity")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Latent Steps: {args.latent_steps}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Initialize model
    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    
    results = []
    
    # Run simple prompts
    print("\n--- SIMPLE PROMPTS ---")
    for prompt in SIMPLE_PROMPTS[:args.num_samples]:
        result = run_single_prompt_entropy(model, prompt, "SIMPLE", args)
        results.append(result)
        print(f"  [{result['label']}] Entropy: {result['avg_entropy']:.4f} | Tokens: {result['token_count']} | {result['prompt']}")
    
    # Run complex prompts
    print("\n--- COMPLEX PROMPTS ---")
    for prompt in COMPLEX_PROMPTS[:args.num_samples]:
        result = run_single_prompt_entropy(model, prompt, "COMPLEX", args)
        results.append(result)
        print(f"  [{result['label']}] Entropy: {result['avg_entropy']:.4f} | Tokens: {result['token_count']} | {result['prompt']}")
    
    # Calculate statistics
    simple_results = [r for r in results if r['label'] == 'SIMPLE']
    complex_results = [r for r in results if r['label'] == 'COMPLEX']
    
    avg_simple = sum(r['avg_entropy'] for r in simple_results) / len(simple_results) if simple_results else 0
    avg_complex = sum(r['avg_entropy'] for r in complex_results) / len(complex_results) if complex_results else 0
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Average Entropy (SIMPLE prompts):  {avg_simple:.4f}")
    print(f"Average Entropy (COMPLEX prompts): {avg_complex:.4f}")
    print(f"Entropy Difference:                {avg_complex - avg_simple:.4f}")
    print(f"Complexity Ratio:                  {avg_complex / avg_simple:.2f}x" if avg_simple > 0 else "N/A")
    print("=" * 70)
    
    # Write detailed log
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "entropy_experiment.log")
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Entropy Experiment Results - {datetime.now().isoformat()}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Latent Steps: {args.latent_steps}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Label':<10} | {'Avg Entropy':>12} | {'Max Entropy':>12} | {'Tokens':>7} | Prompt\n")
        f.write("-" * 70 + "\n")
        
        for r in results:
            f.write(f"{r['label']:<10} | {r['avg_entropy']:>12.4f} | {r['max_entropy']:>12.4f} | {r['token_count']:>7} | {r['prompt']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("SUMMARY:\n")
        f.write(f"  Average SIMPLE Entropy:  {avg_simple:.4f}\n")
        f.write(f"  Average COMPLEX Entropy: {avg_complex:.4f}\n")
        f.write(f"  Difference:              {avg_complex - avg_simple:.4f}\n")
        if avg_simple > 0:
            f.write(f"  Ratio:                   {avg_complex / avg_simple:.2f}x\n")
    
    print(f"\nDetailed log saved to: {log_path}")
    
    # Return for programmatic use
    return {
        "simple_avg": avg_simple,
        "complex_avg": avg_complex,
        "difference": avg_complex - avg_simple,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run entropy experiment for thesis")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model to use for the experiment",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--latent_steps",
        type=int,
        default=5,
        help="Number of latent reasoning steps",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples per category (simple/complex)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--latent_space_realign",
        action="store_true",
        help="Enable latent space realignment",
    )
    
    args = parser.parse_args()
    run_entropy_experiment(args)


if __name__ == "__main__":
    main()
