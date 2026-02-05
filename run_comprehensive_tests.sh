#!/bin/bash
# ============================================================================
# COMPREHENSIVE TEST SUITE FOR Q-STITCH THESIS
# ============================================================================
# This script collects comprehensive data for your thesis:
# 1. Entropy measurements across models and prompts
# 2. Quantization comparison (2, 4, 8, 16-bit) with more samples
# 3. Method comparison (baseline vs latent_mas)
# 4. Multi-task evaluation (GSM8K, ARC, etc.)
# ============================================================================

set -e  # Exit on error

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/comprehensive_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================================================"
echo "Q-STITCH COMPREHENSIVE TEST SUITE"
echo "Started at: $(date)"
echo "Log directory: $LOG_DIR"
echo "============================================================================"

# ============================================================================
# TEST 1: ENTROPY EXPERIMENTS WITH MULTIPLE MODELS
# ============================================================================
echo ""
echo "[TEST 1] ENTROPY EXPERIMENTS - Multiple Models"
echo "============================================================================"

# Test with Qwen2.5-0.5B (baseline, fast)
echo "Running entropy test with Qwen2.5-0.5B..."
python run_entropy_experiment.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --latent_steps 10 \
    2>&1 | tee "$LOG_DIR/entropy_0.5B.log"

# Test with Qwen2.5-1.5B (medium model)
echo "Running entropy test with Qwen2.5-1.5B..."
python run_entropy_experiment.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --latent_steps 10 \
    2>&1 | tee "$LOG_DIR/entropy_1.5B.log"

# Test with Qwen2.5-3B (larger model - key for thesis)
echo "Running entropy test with Qwen2.5-3B..."
python run_entropy_experiment.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --latent_steps 10 \
    2>&1 | tee "$LOG_DIR/entropy_3B.log"

# ============================================================================
# TEST 2: GSM8K BENCHMARK - QUANTIZATION COMPARISON
# ============================================================================
echo ""
echo "[TEST 2] GSM8K QUANTIZATION COMPARISON"
echo "============================================================================"

N_SAMPLES=50  # 50 samples for statistical significance

# 16-bit baseline
echo "Running GSM8K with 16-bit (baseline)..."
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --task gsm8k \
    --max_samples $N_SAMPLES \
    --latent_steps 5 \
    --quant_bits 16 \
    2>&1 | tee "$LOG_DIR/gsm8k_1.5B_16bit.log"

# 8-bit
echo "Running GSM8K with 8-bit..."
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --task gsm8k \
    --max_samples $N_SAMPLES \
    --latent_steps 5 \
    --quant_bits 8 \
    2>&1 | tee "$LOG_DIR/gsm8k_1.5B_8bit.log"

# 4-bit (your sweet spot - 75% savings)
echo "Running GSM8K with 4-bit..."
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --task gsm8k \
    --max_samples $N_SAMPLES \
    --latent_steps 5 \
    --quant_bits 4 \
    2>&1 | tee "$LOG_DIR/gsm8k_1.5B_4bit.log"

# 2-bit (aggressive compression)
echo "Running GSM8K with 2-bit..."
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --task gsm8k \
    --max_samples $N_SAMPLES \
    --latent_steps 5 \
    --quant_bits 2 \
    2>&1 | tee "$LOG_DIR/gsm8k_1.5B_2bit.log"

# ============================================================================
# TEST 3: METHOD COMPARISON (baseline vs latent_mas)
# ============================================================================
echo ""
echo "[TEST 3] METHOD COMPARISON"
echo "============================================================================"

# Baseline (no latent steps)
echo "Running baseline method..."
python run.py \
    --method baseline \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --task gsm8k \
    --max_samples $N_SAMPLES \
    2>&1 | tee "$LOG_DIR/method_baseline.log"

# LatentMAS with different latent steps
for STEPS in 3 5 10; do
    echo "Running latent_mas with $STEPS steps..."
    python run.py \
        --method latent_mas \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --task gsm8k \
        --max_samples $N_SAMPLES \
        --latent_steps $STEPS \
        --quant_bits 4 \
        2>&1 | tee "$LOG_DIR/method_latent_${STEPS}steps.log"
done

# ============================================================================
# TEST 4: LARGER MODEL TEST (Qwen2.5-3B on GSM8K)
# ============================================================================
echo ""
echo "[TEST 4] LARGER MODEL (3B) ON GSM8K"
echo "============================================================================"

echo "Running Qwen2.5-3B on GSM8K (4-bit)..."
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --task gsm8k \
    --max_samples 30 \
    --latent_steps 5 \
    --quant_bits 4 \
    2>&1 | tee "$LOG_DIR/gsm8k_3B_4bit.log"

# ============================================================================
# TEST 5: MULTI-TASK EVALUATION (Different benchmarks)
# ============================================================================
echo ""
echo "[TEST 5] MULTI-TASK EVALUATION"
echo "============================================================================"

# ARC-Easy (reasoning)
echo "Running ARC-Easy..."
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --task arc_easy \
    --max_samples 30 \
    --latent_steps 5 \
    --quant_bits 4 \
    2>&1 | tee "$LOG_DIR/arc_easy_1.5B_4bit.log"

# ARC-Challenge (harder reasoning)
echo "Running ARC-Challenge..."
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --task arc_challenge \
    --max_samples 30 \
    --latent_steps 5 \
    --quant_bits 4 \
    2>&1 | tee "$LOG_DIR/arc_challenge_1.5B_4bit.log"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================================================"
echo "ALL TESTS COMPLETED!"
echo "Finished at: $(date)"
echo "Results saved in: $LOG_DIR"
echo "============================================================================"
echo ""
echo "Log files created:"
ls -la "$LOG_DIR"
