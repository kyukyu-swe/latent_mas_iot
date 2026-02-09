#!/usr/bin/env bash
# Q-Stitch paper experiments: one script to run the full matrix on RunPod.
# Edit the variables below, then: chmod +x run_paper_experiments.sh && ./run_paper_experiments.sh
# Or run inside tmux: tmux new -s paper && ./run_paper_experiments.sh

set -e
cd "$(dirname "$0")"

# --- EDIT THESE (use 3B/7B for paper-ready results) ---
export MODEL="Qwen/Qwen2.5-3B-Instruct"
export AGENTS="Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-3B-Instruct"
export N=100
export LATENT_STEPS=4
export TASK=gsm8k
# For 7B on 24GB GPU, set GENERATE_BS=4 below and add it to run.py calls
export GENERATE_BS=8
# --- END EDIT ---

export LOGDIR="logs/paper_$(date +%Y%m%d)"
mkdir -p "$LOGDIR"
echo "Logs in $LOGDIR"

run() {
  local name="$1"
  shift
  echo ">>> $name"
  python run.py --generate_bs "$GENERATE_BS" "$@" 2>&1 | tee "$LOGDIR/${name}.log"
}

# 1. Baseline
run "baseline_${TASK}_n${N}" \
  --method baseline --model_name "$MODEL" --task "$TASK" --max_samples $N --seed 42

# 2. Text-MAS
run "text_mas_${TASK}_n${N}" \
  --method text_mas --model_name "$MODEL" --task "$TASK" --max_samples $N --seed 42 --prompt sequential

# 3. Latent hybrid 16-bit
run "latent_hybrid_16bit_${TASK}_n${N}" \
  --method latent_mas_hybrid --model_name "$MODEL" --agent_models $AGENTS \
  --task "$TASK" --max_samples $N --seed 42 --prompt sequential \
  --latent_steps $LATENT_STEPS --quant_bits 16 --latent_space_realign

# 4. Quantization comparison (16/8/4/2-bit)
run "compare_quant_${TASK}_n${N}" \
  --method latent_mas_hybrid --model_name "$MODEL" --agent_models $AGENTS \
  --task "$TASK" --max_samples $N --seed 42 --prompt sequential \
  --latent_steps $LATENT_STEPS --latent_space_realign --compare_quantizations
cp -n logs/quantization_comparison.log "$LOGDIR/" 2>/dev/null || true

# 5. EGSQ adaptive
run "latent_hybrid_adaptive_${TASK}_n${N}" \
  --method latent_mas_hybrid --model_name "$MODEL" --agent_models $AGENTS \
  --task "$TASK" --max_samples $N --seed 42 --prompt sequential \
  --latent_steps $LATENT_STEPS --latent_space_realign --adaptive_sieve

echo "Done. Check $LOGDIR and logs/quantization_comparison.log"
