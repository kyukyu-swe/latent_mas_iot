import argparse
import json
from typing import Dict, List, Tuple

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa,
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.latent_mas_hybrid import LatentMASMethod as LatentMASHybridMethod
from methods.text_mas import TextMASMethod
from models import (
    ModelWrapper,
    set_quantization_stats_collector,
    get_and_reset_quantization_stats,
)
from utils import auto_device, set_seed
import time
import os


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


# Main processing function for each batch
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm:
        results = method.run_batch_vllm(current_batch)
    else:
        results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(
            f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}"
        )

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument(
        "--method",
        choices=["baseline", "text_mas", "latent_mas", "latent_mas_hybrid"],
        required=True,
        help="Which multi-agent method to run: 'baseline', 'text_mas', 'latent_mas', or 'latent_mas_hybrid'.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to use (e.g. 'Qwen/Qwen3-8B', 'Qwen/Qwen2.5-1.5B-Instruct', etc.)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Number of questions to evaluate; set -1 to use all samples.",
    )
    parser.add_argument(
        "--task",
        choices=[
            "gsm8k",
            "aime2024",
            "aime2025",
            "gpqa",
            "arc_easy",
            "arc_challenge",
            "mbppplus",
            "humanevalplus",
            "medqa",
        ],
        default="gsm8k",
        help="Dataset/task to evaluate. Controls which loader is used.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=["sequential", "hierarchical"],
        default="sequential",
        help="Multi-agent system architecture: 'sequential' or 'hierarchical'.",
    )

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument(
        "--latent_steps",
        type=int,
        default=0,
        help="Number of latent steps for LatentMAS method",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--generate_bs", type=int, default=20, help="Batch size for generation"
    )
    parser.add_argument(
        "--text_mas_context_length",
        type=int,
        default=-1,
        help="TextMAS context length limit",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Manually add think token in the prompt for LatentMAS",
    )
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # vLLM support
    parser.add_argument(
        "--use_vllm", action="store_true", help="Use vLLM backend for generation"
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        help="Enable prefix caching in vLLM for latent_mas",
    )
    parser.add_argument(
        "--use_second_HF_model",
        action="store_true",
        help="Use a second HF model for latent generation in latent_mas",
    )
    parser.add_argument(
        "--device2",
        type=str,
        default=None,
        help="Second device for HF model (defaults to same as --device)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="How many GPUs vLLM should shard the model across",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Target GPU memory utilization for vLLM",
    )

    # Hybrid method arguments
    parser.add_argument(
        "--agent_models",
        type=str,
        nargs="+",
        default=None,
        help="List of models for each agent in hybrid mode (e.g., 'Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3-8B Qwen/Qwen2.5-0.5B-Instruct')",
    )

    # quantization arguments
    parser.add_argument(
        "--quant_bits",
        type=int,
        default=16,
        help="Compression level for latent thoughts. Use 2, 4, 8, or 16.",
    )
    parser.add_argument(
        "--compare_quantizations",
        action="store_true",
        help="Run once for 16-, 8-, 4-, and 2-bit and write one comparison table to logs/quantization_comparison.log",
    )

    args = parser.parse_args()

    # Default device2 to device if not specified
    if args.device2 is None:
        args.device2 = args.device

    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True
        args.enable_prefix_caching = True

    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)

    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args,
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == "latent_mas":
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == "latent_mas_hybrid":
        method = LatentMASHybridMethod(
            model,
            agent_models=args.agent_models,  # Can be None (same model) or list of models
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []

    # dataset loading
    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split="train")
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split="test")
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split="test")
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split="test")
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split="test")
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split="test")
    elif args.task == "medqa":
        dataset_iter = load_medqa(split="test")
    else:
        raise ValueError(f"no {args.task} support")

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)
        args.max_samples = len(dataset_iter)

    # When comparing quantizations, we need to iterate 3 times over the same data
    if getattr(args, "compare_quantizations", False):
        dataset_iter = list(dataset_iter)

    # Optional: run 16-, 8-, 4-bit in one go and write a single comparison table
    if getattr(args, "compare_quantizations", False) and args.method in (
        "latent_mas_hybrid",
        "latent_mas",
    ):
        results_rows = []
        for quant_bits in [16, 8, 4, 2]:
            args.quant_bits = quant_bits
            set_quantization_stats_collector(
                {"total_transmitted_mb": 0.0, "num_steps": 0}
            )
            preds_q: List[Dict] = []
            processed_q = 0
            batch_q: List[Dict] = []
            run_start = time.time()
            progress_q = tqdm(total=args.max_samples, desc=f"quant_bits={quant_bits}")
            for item in dataset_iter:
                if processed_q >= args.max_samples:
                    break
                batch_q.append(item)
                if (
                    len(batch_q) == args.generate_bs
                    or processed_q + len(batch_q) == args.max_samples
                ):
                    processed_q, preds_q = process_batch(
                        method,
                        batch_q,
                        processed_q,
                        preds_q,
                        progress_q,
                        args.max_samples,
                        args,
                    )
                    batch_q = []
                    if processed_q >= args.max_samples:
                        break
            if batch_q and processed_q < args.max_samples:
                processed_q, preds_q = process_batch(
                    method,
                    batch_q,
                    processed_q,
                    preds_q,
                    progress_q,
                    max_samples=args.max_samples,
                    args=args,
                )
            progress_q.close()
            total_time_q = time.time() - run_start
            acc_q, correct_q = evaluate(preds_q)
            total_mb, num_steps = get_and_reset_quantization_stats()
            bandwidth_per_step = total_mb / num_steps if num_steps > 0 else 0.0
            results_rows.append(
                (quant_bits, bandwidth_per_step, acc_q, total_time_q, correct_q)
            )

        baseline_bw = results_rows[0][1] if results_rows else 0.0
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        table_path = os.path.join(log_dir, "quantization_comparison.log")
        n_samples = args.max_samples
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(
                "Quantization | Bandwidth (MB/Step) | Bandwidth Saving | "
                f"Accuracy (n={n_samples}) | Total Time (s)\n"
            )
            f.write("-" * 80 + "\n")
            for q, bw, acc, tt, correct in results_rows:
                saving = (1 - bw / baseline_bw) * 100 if baseline_bw > 0 else 0.0
                label = "16-bit (Baseline)" if q == 16 else f"{q}-bit"
                f.write(f"{label} | {bw:.4f} | {saving:.1f}% | {acc:.2f} | {tt:.2f}\n")
        with open(table_path, "r", encoding="utf-8") as f:
            print("\n" + f.read())
        print(f"Table saved to {table_path}")
        return

    progress = tqdm(total=args.max_samples)

    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                args.max_samples,
                args,
            )
            batch = []
            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=args.max_samples,
            args=args,
        )
    progress.close()

    total_time = time.time() - start_time

    acc, correct = evaluate(preds)

    # Load results in JSON format
    print(
        json.dumps(
            {
                "method": args.method,
                "model": args.model_name,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "accuracy": acc,
                "correct": correct,
                "total_time_sec": round(total_time, 4),
                "time_per_sample_sec": round(total_time / args.max_samples, 4),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
