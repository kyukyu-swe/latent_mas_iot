<a name="readme-top"></a>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo.png">
    <img alt="LatentMAS" src="assets/logo.png" width=500>
  </picture>
</p>

<h3 align="center">
Latent Collaboration in Multi-Agent Systems
</h3>



<p align="center">
    <a href="https://arxiv.org/abs/2511.20639"><img src="https://img.shields.io/badge/arXiv-2511.20639-B31B1B.svg?logo=arxiv" alt="Arxiv"></a>
    <a href="https://huggingface.co/papers/2511.20639"><img src="https://img.shields.io/badge/Huggingface-DailyPaper-FFD21E.svg?logo=huggingface" alt="Huggingface Paper"></a>
    <a href="https://x.com/LingYang_PU/status/1993510834245714001"><img src="https://img.shields.io/badge/Coverage-LatentMAS-2176BC.svg?logo=x" alt="X"></a>
  
  </p>

---

<p align="center">
  <img src="assets/main_res.png" width="1000">
</p>

# ⚡️ LatentMAS-Hybrid (Fork)

> **Note:** This is a fork of the original [LatentMAS](https://github.com/Gen-Verse/LatentMAS) repository.

## 🌟 Contribution: Heterogeneous Latent Communication

This fork extends the framework to support **Hybrid Multi-Agent Systems**, where agents can use **different model checkpoints** (from the same family) while still communicating via latent representations.

### Motivation
The original homogeneous setting resembles recursive Chain-of-Thought (CoT). The true potential of MAS lies in **specialization**—leveraging large, capable models for high-level planning and smaller, faster models for execution. This hybrid approach enables efficient collaboration without forcing a single model size for all tasks.

### Method: Cross-Model Alignment
To enable latent communication between different models without training adapters, we extend the linear alignment mechanism. We align Model A's output to Model B's input space via their shared vocabulary.

**The Math:**
The original alignment (Eq. 8) minimizes $\|W_{out} W_a - W_{in}\|_F^2$.
For cross-model transfer, we solve for a transformation matrix $W_{cross}$ that maps Model A's hidden states to Model B's input embeddings:

$$W_{cross} = (W_{out,A}^T W_{out,A} + \lambda I)^{-1} W_{out,A}^T W_{in,B}$$

This effectively maps the latent state $h_A$ to the embedding in Model B that corresponds to the same semantic concept.

### Usage
Use the `latent_mas_hybrid` method and specify `--agent_models`:

```bash
python run.py \
  --method latent_mas_hybrid \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --agent_models Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
  --task gsm8k \
  --prompt sequential
```

---

## 💡 Introduction


**LatentMAS** is a multi-agent reasoning framework that **moves agent collaboration from token space into the model’s latent space**.  
Instead of producing long textual reasoning traces, agents communicate by **passing latent thoughts** through their own **working memory**. LatentMAS has the following key features:

- **Efficient** multi-step reasoning with drastically fewer tokens  
- **Training-free** latent-space alignment for stable generation  
- **A general technique** compatible with **any HF model** and optionally **vLLM** backends.

Overall, LatentMAS achieves **superior performance**, **lower token usage**, and **major wall-clock speedups** of multi-agent system.

<p align="center">
  <img src="assets/main.png" width="1000">
</p>


## 🔔 News

- **[2025-11-25]** We have released our paper and code implementations for LatentMAS! Stay tuned for more model-backbone supports and advanced features!
- **[2025-11-25]** We are featured as 🤗 [**HuggingFace 1st Paper of the Day**](https://huggingface.co/papers/2511.20639)!

## 📊 Experiments Overview


### ⭐ Main Results  
Three main tables from our paper spanning 9 tasks across math & science reasoning, commensonse reasoning, and code generation:

- **Table 1 — LatentMAS under the Sequantial MAS setting**  
  <p align="center"><img src="assets/main_table1.png" width="1000"></p>

- **Table 2 — LatentMAS under the Hierarchical MAS setting**  
  <p align="center"><img src="assets/main_table2.png" width="1000"></p>

- **Table 3 — Main Results on Reasoning Intensive Tasks**
  <p align="center"><img src="assets/main_table3.png" width="1000"></p>


### ⚡ Superior Efficiency on **Time and Tokens**

Overall, LatentMAS reduces:
- **~50–80% tokens**
- **~3×–7× wall-clock time**
compared to standard Text-MAS or chain-of-thought baselines.


## 🛠️ Getting Started

This repository provides all code for reproducing LatentMAS, TextMAS, and baseline single-agent experiments across GSM8K, AIME24/25, GPQA, ARC-Easy/Challenge, MBPP+, HumanEval+, and MedQA.

### ⚙️ Setup Environment Variables

We recommend setting your HF cache directory to avoid repeated downloads:

```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
````

Models and datasets will automatically be downloaded into `$HF_HOME`.


### 📦 Install Packages

```bash
conda create -n latentmas python=3.10 -y
conda activate latentmas

pip install -r requirements.txt
```

If you want **vLLM support**, also install:

```bash
pip install vllm
```

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Gen-Verse/LatentMAS.git
cd LatentMAS
```

### 2. Repository Structure

```
LatentMAS/
│── run.py                 # Main entry for experiments
│── models.py              # Wrapper for HF + vLLM + latent realignment
│── methods/
│   ├── baseline.py        # Single-agent baseline
│   ├── text_mas.py        # Token-space multi-agent method
│   └── latent_mas.py      # Latent-space multi-agent (our method)
│── prompts.py             # Prompt constructors
│── data.py                # Dataset loaders
│── data/                  # Provided data + figures (We give medqa.json as an example here)
│── utils.py               # Answer parsing / timeout / helpers
│── example_logs/          # Example logs from LatentMAS
│── requirements.txt
```


## 🧪 Running Experiments (standard HF backend)

### 🔹 **Baseline (single model)**

```bash
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k --max_samples -1 --max_new_tokens 2048
```


### 🔹 **TextMAS (text based multi-agent system)**

```bash
python run.py --method text_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048
```


### 🔹 **LatentMAS (our latent mas method)**

```bash
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048
```

#### Notes:

* **`--latent_steps`** ∈ [0, 80]
  Tune for best performance.
* **`--latent_space_realign`**
  Enables latent→embedding alignment
  We treat this as a **hyperparameter** — enable/disable depending on task/model:

```bash
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --latent_space_realign --max_new_tokens 2048
```


## 📘 Example Logs

Two example LatentMAS logs are provided for reference purposes:

* `example_logs/qwen3_14b_mbppplus_sequential.txt`
* `example_logs/qwen3_14b_humanevalplus_hierarchical.txt`


Please refer to additional experiment logs [here](https://drive.google.com/drive/folders/1evGv5YAmLb4YM_D9Yu0ABa1nfqHC5N-l?usp=drive_link).
You can open them to view the full agent interaction traces and outputs.


## ⚡ vLLM Integration

LatentMAS supports vLLM for faster inference.

### 🔹 Baseline with vLLM

```bash
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k --max_samples -1 --use_vllm --max_new_tokens 2048
```

### 🔹 TextMAS with vLLM

```bash
python run.py --method text_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --use_vllm --max_new_tokens 2048
```

### 🔹 LatentMAS with vLLM

LatentMAS supports a **hybrid HF + vLLM pipeline** for fast inference:
- vLLM handles **final text generation** (with prefix caching, tensor parallelism, etc.)
- A HuggingFace model handles **latent-space rollout** and hidden-state alignment

For this setup, we recommend using two GPUs:
- One GPU for vLLM (`--device`, e.g., `cuda:0`)
- One GPU for the auxiliary HF model (`--device2`, e.g., `cuda:1`)

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048 \
  --use_vllm \
  --use_second_HF_model \
  --enable_prefix_caching \
  --device2 cuda:1
```

**📍Important Note:**

> vLLM does **not** officially support modifying KV-cache or prompting via latent embeddings.
> We modify the partial inner package inside vLLM backend for our method implementation.
> Note minor numeric differences may arise compared to offical HF backend due to different decoding (generation) strategies. Please Use the HF backend to reproduce the official published results.

## 🌐 Awesome Works based on LatentMAS

1. KNN-LatentMAS: [Blog](https://bookmaster9.github.io/kNN-latentMAS/) and [Code](https://github.com/Bookmaster9/kNN-latentMAS).

## 📚 Citation

💫 If you find **LatentMAS** helpful, please kindly give us a star ⭐️ and cite below. Thanks!

```
@article{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and Li, Gaotang and Tieu, Katherine and Lu, Pan and Shen, Ke and Tong, Hanghang and Choi, Yejin and He, Jingrui and Zou, James and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2511.20639},
  year={2025}
}
```

## 🤝 Ackowledgement 

This code is partially based on the amazing work of [vLLM](https://github.com/vllm-project/vllm).
