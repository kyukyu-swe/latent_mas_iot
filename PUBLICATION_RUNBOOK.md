# Q-Stitch: Publication Runbook (Step-by-Step to Submission)

**Goal:** Collect publication-ready results on RunPod, then write and submit the paper.  
**Principle:** No 0.5B/1B-only results as main evidence. Use 1.5B+ and, if GPU allows, 3B/7B for hybrid.

Use this as your checklist. Do steps in order. When a step is done, check it off and move to the next.

---

## Part A: Lock the experimental design (do this first)

### A.1 Publication-ready model choice

**Recommendation: use 3B and 7B for the paper.** They look credible to reviewers; 1.5B is fine as an extra ablation. Qwen2.5 has 3B and 7B (no 4B in the lineup).

| GPU setup | Recommended config | Example `--model_name` / `--agent_models` |
|-----------|--------------------|-------------------------------------------|
| **1× 24GB** | Same-size 3B (faster, fits easily) | `Qwen/Qwen2.5-3B-Instruct` / `3B 3B 3B` |
| **1× 24GB** | Same-size 7B (batch 4–8, may need `--generate_bs 4`) | `Qwen/Qwen2.5-7B-Instruct` / `7B 7B 7B` |
| **1× 40GB or 2× 24GB** | **Heterogeneous 3B + 7B** (strong story) | `Qwen/Qwen2.5-3B-Instruct` / `Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-3B-Instruct` (planner 7B, critic/refiner 3B) |

- **Do not** use 0.5B (or 1.5B-only) as the *primary* setup for paper tables. You can add 1.5B in ablation or appendix.
- **Stick to Qwen2.5** for main results (3B, 7B); same tokenizer and family keeps comparisons clean.
- **Memory:** 7B ≈ 14–16GB bf16; 3B ≈ 6GB. Hybrid with 7B+3B on one GPU is tight on 24GB (use `--generate_bs 2` or 4); two GPUs or 40GB is comfortable.

### A.2 Tasks and sample sizes

| Task | Suggested n (paper) | Notes |
|------|---------------------|--------|
| **GSM8K** | 100 or 200 | Main table; test set is large. |
| **AIME2024 or AIME2025** | 30–50 | Hard math; smaller n is acceptable. |
| **HumanEval+** (optional) | 50–100 | If you want a coding task. |

Start with **GSM8K only** (e.g. 100 samples). Add a second task only after GSM8K tables are done and you have time.

### A.3 Hyperparameters (fixed for all runs)

- `--seed 42`
- `--latent_steps 4` or `8` (pick one and keep it for all latent runs; 4 is faster, 8 often better accuracy)
- `--prompt sequential`
- `--latent_space_realign` (recommended for hybrid)
- `--generate_bs 8` or `10` (smaller if OOM; keep same across runs)

### A.4 Experiment matrix (what to run)

Run these **with the same** `model_name`, `agent_models` (for hybrid), `task`, `max_samples`, `seed`, `latent_steps`.

| # | Method | Extra args | Purpose |
|---|--------|------------|--------|
| 1 | baseline | — | Single-agent baseline |
| 2 | text_mas | — | Text MAS baseline |
| 3 | latent_mas_hybrid | `--quant_bits 16` | Latent hybrid 16-bit (baseline for latent) |
| 4 | latent_mas_hybrid | `--compare_quantizations` | **One script run** → 16/8/4/2-bit table + accuracy + bandwidth |
| 5 | latent_mas_hybrid | `--adaptive_sieve` | EGSQ adaptive |

**Note:** Run 4 gives you a single comparison table. Runs 1–3 and 5 give you accuracy and (where applicable) bandwidth for the main text and tables.

---

## Part B: RunPod setup and sanity check

### B.1 Environment on RunPod

1. Create a pod (e.g. 1× A100 or 1× 4090, 24GB).
2. Clone repo and install:
   ```bash
   cd /workspace  # or your choice
   git clone https://github.com/YOUR_USER/LatentMAS-Hybrid.git   # or upload your code
   cd LatentMAS-Hybrid
   pip install -r requirements.txt
   export HF_HOME=/workspace/hf_cache
   export TRANSFORMERS_CACHE=$HF_HOME
   export HF_DATASETS_CACHE=$HF_HOME
   ```
3. Sanity check (2 samples). **Pick one** depending on your GPU:
   - **3B only (fits 24GB):**
     ```bash
     python run.py --method baseline --model_name Qwen/Qwen2.5-3B-Instruct --task gsm8k --max_samples 2
     ```
   - **7B only (24GB, small batch):**
     ```bash
     python run.py --method baseline --model_name Qwen/Qwen2.5-7B-Instruct --task gsm8k --max_samples 2 --generate_bs 2
     ```
   If the run finishes without OOM, your env is OK.

### B.2 Sanity check for hybrid + quantization

Use **3B** for hybrid sanity (fits 24GB easily). If you have 2 GPUs or 40GB, you can use 7B+3B here instead.

```bash
python run.py --method latent_mas_hybrid \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --agent_models Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-3B-Instruct \
  --task gsm8k --max_samples 2 --latent_steps 4 --quant_bits 8 --latent_space_realign
```

Then one adaptive run:

```bash
python run.py --method latent_mas_hybrid \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --agent_models Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-3B-Instruct \
  --task gsm8k --max_samples 2 --latent_steps 4 --adaptive_sieve --latent_space_realign
```

If both complete, proceed to full runs. For **7B** or **3B+7B** hybrid, use the same commands but swap in `Qwen/Qwen2.5-7B-Instruct` and/or the 7B+3B agent list; use `--generate_bs 2` or `4` if you hit OOM.

**What gets saved when you run the B.2 commands:** Each run writes to its **own** log file (no overwrite):

| Run | Log file path |
|-----|----------------|
| First (fixed 8-bit) | `logs/latent_mas_hybrid/gsm8k_n2_8bit.log` |
| Second (adaptive sieve) | `logs/latent_mas_hybrid/gsm8k_n2_adaptive.log` |

Each file contains: config header, per-problem details, and SUMMARY block (accuracy, correct, total_time_sec, time_per_sample_sec).

---

## Part C: Full runs (copy-paste commands)

Use your chosen **MODEL**, **AGENTS**, **N_SAMPLES**, **LATENT_STEPS** in the commands below.

**Suggested for paper (3B and 7B):**
- **Option A — 3B only (1× 24GB):**  
  `MODEL=Qwen/Qwen2.5-3B-Instruct`  
  `AGENTS="Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-3B-Instruct"`  
  `N=100`, `LATENT_STEPS=4`
- **Option B — 7B only (1× 24GB, smaller batch):**  
  `MODEL=Qwen/Qwen2.5-7B-Instruct`  
  `AGENTS="Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct"`  
  `N=100`, `LATENT_STEPS=4`, add `--generate_bs 4` to every run
- **Option C — Heterogeneous 7B+3B (2× GPU or 40GB):**  
  `MODEL=Qwen/Qwen2.5-3B-Instruct`  
  `AGENTS="Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-3B-Instruct"`  
  `N=100`, `LATENT_STEPS=4`, `--generate_bs 4`

Create a results directory and run from repo root:

```bash
mkdir -p logs/paper_$(date +%Y%m%d)
export LOGDIR=logs/paper_$(date +%Y%m%d)
```

### C.1 Baseline

```bash
python run.py --method baseline \
  --model_name $MODEL --task gsm8k --max_samples $N --seed 42 \
  2>&1 | tee $LOGDIR/baseline_gsm8k_n${N}.log
```

### C.2 Text-MAS

```bash
python run.py --method text_mas \
  --model_name $MODEL --task gsm8k --max_samples $N --seed 42 --prompt sequential \
  2>&1 | tee $LOGDIR/text_mas_gsm8k_n${N}.log
```

### C.3 Latent-MAS hybrid 16-bit (latent baseline)

```bash
python run.py --method latent_mas_hybrid \
  --model_name $MODEL --agent_models $AGENTS \
  --task gsm8k --max_samples $N --seed 42 --prompt sequential \
  --latent_steps $LATENT_STEPS --quant_bits 16 --latent_space_realign \
  2>&1 | tee $LOGDIR/latent_hybrid_16bit_gsm8k_n${N}.log
```

### C.4 Quantization comparison (16/8/4/2-bit in one go)

This writes `logs/quantization_comparison.log` and detailed logs per bit-width. Run from repo root so `logs/` is correct.

```bash
python run.py --method latent_mas_hybrid \
  --model_name $MODEL --agent_models $AGENTS \
  --task gsm8k --max_samples $N --seed 42 --prompt sequential \
  --latent_steps $LATENT_STEPS --latent_space_realign --compare_quantizations \
  2>&1 | tee $LOGDIR/compare_quant_gsm8k_n${N}.log
```

After run, copy the comparison table to your paper folder:

```bash
cp logs/quantization_comparison.log $LOGDIR/
```

### C.5 EGSQ adaptive sieve

```bash
python run.py --method latent_mas_hybrid \
  --model_name $MODEL --agent_models $AGENTS \
  --task gsm8k --max_samples $N --seed 42 --prompt sequential \
  --latent_steps $LATENT_STEPS --latent_space_realign --adaptive_sieve \
  2>&1 | tee $LOGDIR/latent_hybrid_adaptive_gsm8k_n${N}.log
```

---

## Part D: Collect and summarize results

### D.1 All runs and their log files (no overwrite)

Every run in this runbook writes to a **distinct** log file. Use the table below to find where each result is saved (replace `100` with your `$N` if different).

| # | Run (section) | Log file path |
|---|----------------|----------------|
| B.1 | Baseline sanity (n=2) | `logs/baseline/gsm8k_n2.log` |
| B.2 | Hybrid 8-bit sanity (n=2) | `logs/latent_mas_hybrid/gsm8k_n2_8bit.log` |
| B.2 | Hybrid adaptive sanity (n=2) | `logs/latent_mas_hybrid/gsm8k_n2_adaptive.log` |
| C.1 | Baseline full | `logs/baseline/gsm8k_n100.log` |
| C.2 | Text-MAS full | `logs/text_mas/gsm8k_n100.log` |
| C.3 | Latent hybrid 16-bit full | `logs/latent_mas_hybrid/gsm8k_n100_16bit.log` |
| C.4 | Compare quantizations (16/8/4/2) | `logs/quantization_16bit_detailed.log`, `logs/quantization_8bit_detailed.log`, `logs/quantization_4bit_detailed.log`, `logs/quantization_2bit_detailed.log`, and `logs/quantization_comparison.log` (table) |
| C.5 | Latent hybrid adaptive full | `logs/latent_mas_hybrid/gsm8k_n100_adaptive.log` |

Each per-method log contains: config header, per-problem details, and **SUMMARY** block (`accuracy`, `correct`, `total`, `total_time_sec`, `time_per_sample_sec`) for Paper Table 1. Stdout also prints a one-line JSON. The comparison run writes the summary table to `logs/quantization_comparison.log`.

### D.2 Paper Table 1 (example)

Fill a table like this from your logs:

| Method | Accuracy (n=100) | Bandwidth (MB/step) | Total time (s) |
|--------|------------------|----------------------|----------------|
| Baseline | … | — | … |
| Text-MAS | … | — | … |
| Latent-MAS hybrid 16-bit | … | … | … |
| Latent-MAS hybrid 8-bit | … | … | … |
| Latent-MAS hybrid 4-bit | … | … | … |
| Latent-MAS hybrid 2-bit | … | … | … |
| Latent-MAS hybrid EGSQ (adaptive) | … | (avg) | … |

Bandwidth and accuracy for 16/8/4/2 come from the comparison run. Baseline, Text-MAS, 16-bit hybrid, and adaptive come from their own runs.

### D.3 Entropy evidence (optional but good for paper)

If you want a figure or table “entropy vs task complexity”:

```bash
python run_entropy_experiment.py --model_name Qwen/Qwen2.5-1.5B-Instruct
```

Use the logged entropy values for simple vs complex prompts. Redirect to a log if needed:

```bash
python run_entropy_experiment.py --model_name Qwen/Qwen2.5-1.5B-Instruct 2>&1 | tee $LOGDIR/entropy_experiment.log
```

---

## Part E: Paper outline and who does what

### E.1 Suggested sections

1. **Abstract** — Problem, method (EGSQ on hybrid latent MAS), main result (bandwidth ↓, accuracy ≈), scope (edge-cloud).
2. **Introduction** — Motivation, gap (no prior entropy-gated adaptive quantization for inter-agent latent channel), contribution (3 bullets).
3. **Related work** — LatentMAS, Hybrid-LatentMAS, quantization (LLM.int8, SmoothQuant, etc.), your positioning (Section 4b/4c in THESIS_ROADMAP.md).
4. **Method** — Pipeline (heterogeneous latent MAS + alignment), fixed quantization, EGSQ (entropy gate, bit-rate rule), optional saliency.
5. **Experiments** — Setup (models, tasks, n, seed), results (Table 1, bandwidth vs accuracy), adaptive vs fixed, entropy vs complexity (if you ran entropy experiment).
6. **Discussion** — Limitations, future work (Jetson, more tasks).
7. **Conclusion** — One paragraph.
8. **Appendix** — Full results, hyperparameters, more ablations if any.

### E.2 Your workflow

- **You:** Run all commands on RunPod, collect logs, fill Table 1 (and Table 2 if second task), write first draft of each section.
- **Partner (AI):** Help with structure, related-work phrasing, “contribution” sentences, figure captions, and tightening methodology. You can paste draft paragraphs and ask for edits.
- **Professor:** High-level feedback and target venue; optional read of intro + experiments when you have a full draft.

---

## Part F: Week-by-week timeline (3 months to submission)

| Week | Focus | Deliverable |
|------|--------|-------------|
| 1 | Lock MODEL, AGENTS, N, LATENT_STEPS. RunPod setup + sanity checks. | Sanity logs; one full comparison run (e.g. n=50) to verify. |
| 2 | Full GSM8K runs (baseline, text_mas, hybrid 16-bit, compare_quantizations, adaptive). N=100. | All logs in `logs/paper_YYYYMMDD/`; Table 1 filled. |
| 3 | (Optional) Second task (AIME or HumanEval+), smaller n. Or replicate GSM8K with different latent_steps. | Table 2 or ablation; entropy experiment log. |
| 4–5 | Write Method + Experiments; paste Table 1 (and 2). Draft Intro + Related work. | Full draft (Method, Experiments, Intro, Related work). |
| 6–8 | Abstract, Discussion, Conclusion; internal pass; figures (e.g. pipeline, EGSQ gate). | Complete draft. |
| 9–10 | Revise from professor feedback (if any); format for target journal; supplement (code, reproducibility). | Submission-ready manuscript + supplement. |
| 11–12 | Submit; prepare rebuttal plan; thesis chapter alignment if needed. | Paper submitted. |

---

## Part G: One-shot run script (optional)

You can put the full run matrix into a shell script so you launch once and get all logs. Example name: `run_paper_experiments.sh`. Set variables at top (MODEL, AGENTS, N, LATENT_STEPS), then run:

- Baseline
- Text-MAS  
- Latent hybrid 16-bit
- Compare quantizations
- Adaptive

Each with `2>&1 | tee $LOGDIR/<name>.log`. Use `nohup` or `tmux` on RunPod so the job continues if you disconnect.

---

## Checklist (mark when done)

- [ ] A.1 Model and agent list fixed (no 0.5B as primary)
- [ ] A.2 Task and n fixed (e.g. GSM8K 100)
- [ ] A.3 latent_steps, seed, prompt fixed
- [ ] B.1 RunPod env installed and sanity run (baseline 2 samples) OK
- [ ] B.2 Hybrid + quant + adaptive sanity (2 samples) OK
- [ ] C.1 Baseline full run done, log saved
- [ ] C.2 Text-MAS full run done, log saved
- [ ] C.3 Latent hybrid 16-bit full run done, log saved
- [ ] C.4 Compare quantizations run done; comparison table copied to $LOGDIR
- [ ] C.5 Adaptive sieve full run done, log saved
- [ ] D.1 Table 1 (and Table 2 if applicable) filled from logs
- [ ] D.2 (Optional) Entropy experiment run and logged
- [ ] E–F Draft written and submission-ready

When you finish a step, you can come back and say “I’m at step C.4” or “Table 1 is done, here are the numbers” and we can do the next part (e.g. writing the Experiments section or the contribution sentence for the abstract).
