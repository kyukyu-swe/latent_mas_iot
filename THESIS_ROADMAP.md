# Q-Stitch: Thesis Roadmap & Proposal

**Project Title:** Q-Stitch — *Adaptive Latent Communication via Entropy-Gated Saliency Quantization for Heterogeneous Edge-Cloud Swarms*

**Institutions:** KMUTT & Kanazawa University  
**Document Purpose:** Formal roadmap linking research objectives, methodology, and codebase for proposal and progress reports.

**Codebase basis:** This project adopts [Hybrid-LatentMAS](https://github.com/nhminle/LatentMAS-Hybrid) (nhminle) as the base implementation and adds Q-Stitch extensions: entropy-gated adaptive quantization (EGSQ), fixed/adaptive bit-width comparison, and bandwidth–accuracy analysis for edge-cloud swarms.

---

## 1. Research Objectives (The "Goals")

| ID | Objective | Success Criterion |
|----|-----------|-------------------|
| **O1** | **Bandwidth Optimization** | Reduce inter-agent communication overhead in MAS by **≥ 75%** using sub-8-bit quantization. |
| **O2** | **Accuracy Preservation** | Maintain **≥ 95%** of baseline (16-bit) reasoning accuracy during heterogeneous model collaboration. |
| **O3** | **Dynamic Adaptation** | Develop an **Information-Aware** algorithm that adjusts bit-precision in real-time based on task complexity. |
| **O4** | **Hardware Validation** | Demonstrate feasibility on resource-constrained Edge hardware (Jetson Nano / IoT devices). |

---

## 2. What Has Been Done (Phase 1: Empirical Validation)

### 2.1 MAS Infrastructure
- **Codebase:** LatentMAS-Hybrid fork with heterogeneous multi-agent support.
- **Location:** `models.py` (ModelWrapper, latent realignment), `methods/latent_mas_hybrid.py` (cross-model transfer).
- **Evidence:** Pipeline allows Llama-3 and Qwen-2.5 (e.g. 7B/8B) to send latent "thoughts" to smaller 1B/3B models via linear alignment \( W_{cross} \).

### 2.2 Quantization Baseline
- **Implementation:** `models.py` → `latent_sieve_quantize(hidden_states, bits)` (fixed 16/8/4/2-bit uniform quantization).
- **Experiments:** `run.py` with `--quant_bits 2|4|8|16` and `--compare_quantizations`.
- **Logs:** `logs/quantization_*bit_detailed.log`, `logs/quantization_comparison.log`.
- **Result:** 4-bit achieves ~75% bandwidth reduction with minimal accuracy loss on Qwen; 2-bit shows "latency cliff" and accuracy drop on Llama-3.

### 2.3 Entropy Discovery (Trigger for EGSQ)
- **Implementation:** `models.py` → `calculate_latent_entropy(hidden_states, lm_head=None)` (Shannon entropy in bits).
- **Probing:** `probe_latent_overhead()` logs **Entropy | Raw MB | Transmitted MB** per step; used in `generate_latent_batch()` and `latent_mas_hybrid.py`.
- **Experiment:** `run_entropy_experiment.py` — correlates entropy with prompt complexity.
- **Result:** Complex tasks (e.g. math) yield ~1.16× higher entropy (~7.0 bits) than simple tasks (e.g. greetings, ~4.5 bits). This justifies **entropy as the gate** for bit-rate selection.

---

## 3. Methodology in Detail: EGSQ Algorithm

### 3.1 Mathematical Model

**Dynamic Gate (vs. static LLM.int8() / SmoothQuant):**

- **Saliency (optional):** Importance of neuron \(i\) by magnitude \( |x_i| \).
- **Entropy gating:** Shannon entropy \( H(X) \) of latent tensor \( X \).
- **Bit-rate rule:**
  \[
  \text{Bit-Rate} = f\bigl(H(X)\bigr)
  \]
  - If \( H(X) > \tau_{\text{high}} \): use **16-bit** (protect complex reasoning).
  - If \( H(X) < \tau_{\text{low}} \): use **2-bit** (maximum efficiency).
  - Else: interpolate (e.g. 4-bit or 8-bit) or step function.

**Implementation (Phase 2):**  
`models.py` → `egsq_adaptive_bits()`, `latent_sieve_quantize_adaptive()`, and `saliency_score_magnitude()` (for future per-neuron use). Called from `generate_latent_batch()` and `methods/latent_mas_hybrid.py` when `--adaptive_sieve` is enabled. CLI: `--adaptive_sieve`, `--entropy_high_threshold`, `--entropy_low_threshold`.

### 3.2 Heterogeneous Stitching

- **Formula:** \( W_{cross} = (W_{out,A}^T W_{out,A} + \lambda I)^{-1} W_{out,A}^T W_{in,B} \)
- **Location:** `methods/latent_mas_hybrid.py` → `transfer_via_realignment()`; `models.py` → `_build_latent_realign_matrix()`, `_apply_latent_realignment()`.
- **Flow:** Cloud model hidden state → (optional EGSQ quantize) → align to Edge model dimension via \( W_{cross} \) → Edge model continues.

---

## 4. Key Literature & Novelty

| Work | Their contribution | Q-Stitch twist |
|------|---------------------|----------------|
| **LLM.int8()** (Dettmers et al., 2022) | Outlier-aware quantization for GPU memory | Apply quantization to **inter-agent communication**, not only local inference. |
| **SmoothQuant** (Xiao et al., 2023) | Static balance of weight/activation outliers | **Entropy-gated dynamic bit-width** per step; SmoothQuant is static. |
| **Neural Stitching** (Pan et al.) | Connecting different models | Add a **quantized communication layer** inside the stitch (EGSQ). |

---

### 4b. Positioning vs Prior Art (Reducing “Others Did It First” Rejection Risk)

Reviewers may ask: “How is this different from X?” Below is the **exact gap** to cite in your related work and intro. Use it to pre-empt rejection.

**Closest prior work:**

| Work | What they do | What they do *not* do (your gap) |
|------|--------------------------|----------------------------------|
| **LatentMAS** (Gen-Verse, 2024–25) | Latent inter-agent communication; may use fixed quantization. | **Heterogeneous** agents (different model sizes) + **adaptive** (entropy-gated) bit-width. They are same-model, and quantization (if any) is fixed. |
| **Interlat** | Latent communication + learned compression. | Compression is **learned**; no **inference-time entropy gate** for bit-width. No heterogeneous alignment. |
| **EdgeQAT** (entropy-guided QAT, 2024) | Entropy/distribution for **quantization-aware training** of one model on edge. | **Inter-agent** channel; **inference-time** gate; no training. Different setting (communication, not single-model weight/activation quant). |
| **AdaQAT** (adaptive bit-width QAT) | Adaptive bit-width **during training**. | We do **inference-time** adaptive bit-width based on **current latent entropy**; no training. |

**One-sentence gap (use in abstract/intro):**  
*“To our knowledge, no prior work combines (1) heterogeneous latent MAS with cross-model alignment, (2) inference-time entropy-gated adaptive quantization of the inter-agent channel, and (3) bandwidth–accuracy tradeoff analysis for edge-cloud swarms.”*

**Suggested related-work phrasing:**  
*“LatentMAS and Interlat show the benefits of latent communication but assume homogeneous agents or fixed compression. Entropy-guided quantization (e.g. EdgeQAT) targets single-model edge deployment with training; we target the **communicated latent** between different-sized agents **at inference time** with a training-free entropy gate.”*

**Before submission (≈ 1 hour):**  
Do a final search for: *“latent multi-agent quantization”*, *“entropy adaptive bit-width communication LLM”*, and *“heterogeneous multi-agent latent”* (2024–2025). If you find something very close, cite it and add one sentence: “Unlike [X], we …” in the intro. That shows rigor and reduces “you missed prior work” rejection.

---

### 4c. Official “Awesome Works” Built on LatentMAS (from Gen-Verse GitHub)

From the [LatentMAS repo](https://github.com/Gen-Verse/LatentMAS) “Awesome Works Built on Top of LatentMAS” (as of 2025). Use this to cite sibling extensions and position Q-Stitch.

| # | Name | Author | What it adds | How Q-Stitch differs |
|---|------|--------|----------------------|----------------------|
| 1 | **Science-LatentMAS** | Prof. Markus J. Buehler & MIT LAMM | Scientific modeling & material-system collaboration; flexible agent types; specialized latent communication for science. | We focus on **bandwidth/quantization** and **edge-cloud**, not science domains. |
| 2 | **KNN-LatentMAS** | Bookmaster9 | kNN-based latent retrieval; better KV-cache usage; memory efficiency and multi-step reasoning stability. | We focus on **entropy-gated adaptive quantization** of the channel and **heterogeneous** alignment, not retrieval/KV. |
| 3 | **Hybrid-LatentMAS** | nhminle | Heterogeneous / hybrid agents (LLM + non-LLM); modular pipelines mixing models, tools, reasoning strategies. [Code: github.com/nhminle/LatentMAS-Hybrid](https://github.com/nhminle/LatentMAS-Hybrid) | **Q-Stitch extends this fork:** we add EGSQ (entropy-gated bit-width), fixed/adaptive quantization, and bandwidth–accuracy analysis for edge. |
| 4 | **Awareness Network** | Everest-AN | Decentralized AI awareness market; autonomous agent collaboration and memory sharing. | We focus on **quantized latent channel** and **single pipeline** bandwidth, not markets/decentralization. |

**For your paper:**  
- Cite **LatentMAS** (base) and **Hybrid-LatentMAS** (nhminle) as the foundation; state that Q-Stitch adds *entropy-gated adaptive quantization and bandwidth–accuracy tradeoffs for edge-cloud swarms* on top of heterogeneous latent MAS.  
- Optionally cite **Science-LatentMAS**, **KNN-LatentMAS**, and **Awareness Network** in related work as other “built on LatentMAS” directions; one sentence each on what they do and that your contribution is orthogonal (quantization/bandwidth/edge).

---

## 5. Phases & Codebase Mapping

| Phase | Goal | Status | Key Files / Commands |
|-------|------|--------|----------------------|
| **Phase 1** | Empirical validation (quantization + entropy) | ✅ Done | `models.py`, `run_entropy_experiment.py`, `run.py --compare_quantizations`, `logs/` |
| **Phase 2** | EGSQ Adaptive Sieve | ✅ Implemented | `models.py` (egsq_adaptive_bits, latent_sieve_quantize_adaptive), `run.py --adaptive_sieve`, `methods/latent_mas_hybrid.py` |
| **Phase 3** | Stress testing | 📋 Planned | HumanEval+, AIME; validate that adaptive sieve avoids 2-bit failure |
| **Phase 4** | Hardware (Jetson Nano) | 📋 Planned | Power (Watts) measurement: 4-bit vs 16-bit "thought" reception |
| **Phase 5** | Writing & submission | 📋 Planned | Target: e.g. IEEE Internet of Things Journal |

---

## 6. File Reference (Academic Rigor)

| File | Role in Q-Stitch |
|------|------------------|
| `models.py` | Entropy computation, fixed & adaptive quantization sieve, latent realignment, probe (bandwidth + entropy). |
| `methods/latent_mas_hybrid.py` | Heterogeneous MAS; transfer via \( W_{cross} \); uses sieve and probe. |
| `run.py` | CLI: `--quant_bits`, `--compare_quantizations`, `--adaptive_sieve`, tasks, logging. |
| `run_entropy_experiment.py` | Thesis evidence: entropy vs. prompt complexity. |
| `prompts.py` | Agent prompts (planner, critic, refiner, judger) for sequential/hierarchical flows. |
| `data.py` | Benchmarks: GSM8K, AIME, ARC, GPQA, MBPP+, HumanEval+, MedQA. |
| `utils.py` | Answer extraction, normalization, timeouts. |
| `README.md` | Setup, usage, hybrid method. |
| `THESIS_ROADMAP.md` | This document: objectives, methodology, phases, file map. |

---

## 7. Reproducibility (For Reviewers)

- **Environment:** `requirements.txt`, Python 3.10, PyTorch, Transformers.
- **Baseline (fixed quantization):**
  ```bash
  python run.py --method latent_mas_hybrid --quant_bits 4 --task gsm8k --max_samples 100
  ```
- **Quantization comparison:**
  ```bash
  python run.py --method latent_mas_hybrid --compare_quantizations --task gsm8k --max_samples 50
  ```
- **Entropy vs. complexity:**
  ```bash
  python run_entropy_experiment.py --model_name Qwen/Qwen2.5-0.5B-Instruct
  ```
- **Adaptive sieve (Phase 2 — EGSQ):**
  ```bash
  python run.py --method latent_mas_hybrid --adaptive_sieve --task gsm8k --max_samples 50
  python run.py --method latent_mas_hybrid --adaptive_sieve --entropy_high_threshold 6.5 --entropy_low_threshold 4.5 --task gsm8k
  ```

---

## 8. Feasibility, Timeline & Publication (3-Month Master’s Plan)

*Practical reality check so the thesis is finishable and defensible without over-promising.*

### Can you finish in 3 months (15 hrs/day) and publish?

- **Finish the Master’s thesis in 3 months:** **Yes, realistic** if you lock scope and prioritize writing. You already have: working system, quantization baseline, entropy evidence, and EGSQ implemented. That is enough for a solid thesis.
- **Publish a journal paper:** Split into two goals:
  - **Submit** a paper (conference or journal) **within or soon after 3 months** — **feasible**. Use the thesis as the draft.
  - **Get accepted** within 3 months — **not realistic**. Review cycles are usually 2–6+ months. Plan to graduate on **thesis defense**, not on “paper accepted.” Publication can follow after graduation.

**Recommendation:** Define success as **“thesis submitted and defended, paper submitted (or under review)”** in 3 months. Acceptance is a bonus.

### What to keep vs defer (scope for feasibility)

| Keep (core of thesis) | Defer or simplify |
|----------------------|-------------------|
| Heterogeneous latent MAS + alignment | Multiple model families at once (stick to Qwen or Llama) |
| Fixed 2/4/8/16-bit baseline + comparison table | Many benchmarks (pick 1–2: e.g. GSM8K + one of AIME/HumanEval+) |
| Entropy probe + “complexity vs entropy” evidence | Extra saliency beyond what you have |
| EGSQ adaptive sieve + ablation (e.g. threshold sensitivity) | Full Jetson deployment (see below) |
| One clear ablation: fixed vs adaptive (accuracy + bandwidth) | “Top-tier” journal as only target (submit to one realistic venue) |
| **O4 Hardware:** Make it **optional** or **proof-of-concept**: e.g. one Jetson run (4-bit vs 16-bit latency/power) OR report GPU latency/memory as “edge-relevant” proxy. Do **not** let hardware block graduation. | |

### Realistic novelty (already there + small additions)

You already have enough novelty for a Master’s thesis:

1. **Entropy-gated dynamic bit-width for inter-agent communication** (not just inference quantization like LLM.int8/SmoothQuant).
2. **Heterogeneous MAS with linear alignment + quantized latent channel** (stitch + sieve).
3. **Empirical evidence:** entropy vs task complexity; 4-bit vs 2-bit tradeoff; adaptive sieve implemented.

**Low-effort, high-value additions** (if time allows):

- **EGSQ threshold ablation:** Run adaptive sieve with 2–3 threshold pairs (e.g. 6.0/4.0, 6.5/4.5, 7.0/5.0); one table in thesis shows sensitivity. (≈ 1–2 days of runs.)
- **One extra task:** Add AIME *or* HumanEval+ (not both) to show generality. (≈ a few days.)
- **“Edge” angle without Jetson:** Report latency (ms) and memory (MB) per step for 2/4/8/16-bit on one GPU as “resource footprint for edge deployment.” (≈ 1 day.)

### Suggested 3-month timeline (high level)

| Window | Focus |
|--------|--------|
| **Month 1** | Lock experiments: GSM8K (+ one more task if planned). Run fixed 2/4/8/16 and adaptive; EGSQ threshold ablation. Lock methodology and related work in thesis draft. |
| **Month 2** | Finish all result tables and figures. Optional: one Jetson or “edge proxy” experiment. Start full thesis write-up (intro, method, experiments, discussion). |
| **Month 3** | Complete thesis draft, internal review, submit to university. Prepare defense. **Submit** paper to one venue (conference or journal); acceptance can come after defense. |

### Where to submit the paper (realistic)

- **Conference:** e.g. IEEE/ACM workshop, regional conference, or application track (faster feedback, good for “first publication”).
- **Journal:** e.g. IEEE Access (open access, relatively faster), or IEEE IoT Journal if you frame it clearly as “efficient multi-agent communication for edge/cloud” — submit when thesis is stable; don’t wait for acceptance to graduate.

### Bottom line

- You **can** finish a **practical, defensible Master’s thesis** in 3 months with 15 hrs/day by: (1) keeping scope tight, (2) treating hardware as optional/proof-of-concept, (3) prioritizing “thesis done + paper submitted” over “paper accepted.”
- Your **technique/novelty is already sufficient**: entropy-gated adaptive quantization for inter-agent latent communication in heterogeneous MAS. Small additions (threshold ablation, one extra benchmark, edge proxy) strengthen the story without risking the timeline.
- Feeling “late” is common; what matters is a **clear, feasible plan**. This section is that plan. Use it with your advisors and adjust only if they require specific additions (e.g. mandatory hardware).

---

*This roadmap ties the thesis proposal to the repository so that KMUTT and Kanazawa reviewers can verify objectives, methodology, and evidence in the codebase.*
