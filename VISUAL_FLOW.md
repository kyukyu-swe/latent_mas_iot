# Hybrid LatentMAS Visual Flow

## Scenario 1: Same Model (KV Cache Sharing)

```
Agent 1 (Qwen-7B)                Agent 2 (Qwen-7B)                Agent 3 (Qwen-7B)
┌──────────────────┐            ┌──────────────────┐            ┌──────────────────┐
│  Prompt + Task   │            │  Prompt + Task   │            │  Prompt + Task   │
└────────┬─────────┘            └────────┬─────────┘            └────────┬─────────┘
         │                               │                               │
         ▼                               ▼                               ▼
┌────────────────────┐          ┌────────────────────┐          ┌────────────────────┐
│ Generate Latent    │──KV──▶  │ Generate Latent    │──KV──▶  │ Generate Answer    │
│ (10 steps)         │  Cache   │ (10 steps)         │  Cache   │                    │
│                    │          │                    │          │                    │
│ 4096-dim hidden    │          │ 4096-dim hidden    │          │ 4096-dim hidden    │
└────────────────────┘          └────────────────────┘          └────────────────────┘
         │                               │                               │
         │ Fast! Just pointer            │                               │
         │ No transfer needed            │                               │
         ▼                               ▼                               ▼
  [KV Cache State]              [KV Cache State]                [Final Answer]
  • Same model                  • Accumulates                   
  • Same dimensions             • Efficient                     
```

**Key**: KV cache is a tuple of (keys, values) for each layer. Just passes through!

---

## Scenario 2: Different Models (Vocabulary-Space Transfer)

```
Agent 1: Solver                Transfer 0.5B→7B              Agent 2: Reviewer
(Qwen-0.5B: 896 dims)                                        (Qwen-7B: 4096 dims)
┌──────────────────┐                                        ┌──────────────────┐
│  Solve Problem   │                                        │  Review Solution │
└────────┬─────────┘                                        └────────┬─────────┘
         │                                                           │
         ▼                                                           ▼
┌────────────────────┐                                      ┌────────────────────┐
│ Generate Latent    │                                      │ Receive Transfer   │
│ (10 steps)         │                                      │ Embeddings         │
│                    │                                      │                    │
│ 896-dim hidden ────┼──────────────────────────────────▶  │ 4096-dim embeds    │
└────────────────────┘                                      └──────────┬─────────┘
         │                                                           │
         │ Extract hidden states (PRE-realignment)                  │
         │ [batch, seq, 896]                                        │
         ▼                                                           ▼
┌─────────────────────────────────────────────────────────┐  ┌──────────────────┐
│              VOCABULARY TRANSFER                         │  │ Generate Latent  │
│                                                          │  │ (10 steps)       │
│  Step 1: hidden @ W_out_0.5B^+  →  vocab_logits         │  │                  │
│          [batch,seq,896] @ [896,152K] = [batch,seq,152K]│  │ 4096-dim hidden  │
│                                                          │  └──────────┬───────┘
│  Step 2: vocab_logits @ W_in_7B  →  embeddings          │           │
│          [batch,seq,152K] @ [152K,4096] = [batch,seq,4096]│          │
│                                                          │           ▼
│  Vocabulary = semantic common ground (same tokenizer)   │  [Extract for next]
└─────────────────────────────────────────────────────────┘
         │
         │ Fresh embeddings in 7B's space
         │ 7B will apply ITS OWN realignment
         ▼
Transfer 7B→1.5B              Agent 3: Judger
                              (Qwen-1.5B: 1536 dims)
                              ┌──────────────────┐
                              │  Final Decision  │
                              └────────┬─────────┘
                                       ▼
                              ┌──────────────────┐
                              │ Generate Answer  │
                              │                  │
                              │ 1536-dim final   │
                              └──────────────────┘
```

**Key**: Each transfer goes through vocabulary space (152K tokens) which provides semantic grounding.

---

## Critical Detail: Pre-Realignment Transfer

```
Agent A (Model A)
├─ Process input
├─ Hidden states [batch, seq, dim_A]     ◀─── Extract these
└─ (Skip realignment for transfer)
    │
    ├─ Transfer RAW hidden states
    │   (no model-specific interpretation yet)
    │
    ▼
Vocabulary Transfer
├─ hidden_A @ W_out_A^+ → vocab_logits   (A's vocab interpretation)
├─ vocab_logits @ W_in_B → embedding_B   (B's fresh embeddings)
└─ Embeddings in B's space, no prior realignment
    │
    ▼
Agent B (Model B)
├─ Process embeddings_B
├─ Hidden states_B [batch, seq, dim_B]
└─ Apply realignment_B        ◀─── CORRECT! B applies ITS OWN realignment
    (B interprets in its own space)
```

**Why this matters**:
- Realignment matrix: `M = (W_out^T W_out)^-1 W_out^T W_in`
- Model-specific! 0.5B's M ≠ 7B's M
- Each model should interpret outputs in its own embedding space
- Vocabulary provides neutral semantic transfer medium

---

## Data Flow Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid LatentMAS Flow                         │
└─────────────────────────────────────────────────────────────────┘

Input: "What is 15 + 27?"

Agent 1: Solver (Qwen-0.5B, 896 dims)
  ├─ Tokenize: [15, 373, 374] (example tokens)
  ├─ Embed: W_in_0.5B[tokens] → [batch, 3, 896]
  ├─ Process through transformer
  ├─ Generate latent thoughts (10 steps)
  │   └─ Each step: hidden → realign → embed → repeat
  ├─ Final hidden state: [batch, 13, 896] (3 input + 10 latent)
  └─ Extract: hidden_states (NO realignment applied)

Model Switch Detected! 0.5B → 7B

Transfer via Vocabulary Space:
  ├─ Compute W_out_0.5B^+ (pseudoinverse)
  ├─ hidden[896] @ W_out_0.5B^+[896→152K] = vocab_logits[152K]
  │   └─ Interpretation: "These hidden states mean tokens [1, 5, 42, 99, ...]"
  ├─ vocab_logits[152K] @ W_in_7B[152K→4096] = embeddings_7B[4096]
  │   └─ "Re-embed those tokens in 7B's space"
  └─ Result: Fresh embeddings [batch, 13, 4096]

Agent 2: Reviewer (Qwen-7B, 4096 dims)
  ├─ Receive: embeddings_7B [batch, 13, 4096]
  ├─ Create KV cache from these embeddings
  ├─ Process new prompt: "Review the solution"
  ├─ Generate latent thoughts (10 steps)
  │   └─ 7B applies ITS OWN realignment to ITS OWN hidden states
  ├─ Final hidden state: [batch, 13+prompt+10, 4096]
  └─ Extract: hidden_states (NO realignment applied)

Model Switch Detected! 7B → 1.5B

Transfer via Vocabulary Space:
  ├─ hidden[4096] @ W_out_7B^+[4096→152K] = vocab_logits[152K]
  ├─ vocab_logits[152K] @ W_in_1.5B[152K→1536] = embeddings_1.5B[1536]
  └─ Result: Fresh embeddings [batch, ~23, 1536]

Agent 3: Judger (Qwen-1.5B, 1536 dims)
  ├─ Receive: embeddings_1.5B [batch, ~23, 1536]
  ├─ Create KV cache from these embeddings
  ├─ Process final prompt: "Provide final answer"
  ├─ Generate answer (NOT latent thoughts)
  │   └─ Standard text generation with KV cache
  └─ Output: "42"

Final Answer: "42"
```

---

## Memory and Efficiency Comparison

```
┌────────────────────────────────────────────────────────────────┐
│            Same Model (KV Cache)                                │
├────────────────────────────────────────────────────────────────┤
│ Memory: 1x model (e.g., 7B params ≈ 14GB)                      │
│ Transfer: None! Just pointer passing                            │
│ Latency: ~0ms transfer overhead                                 │
│ Use case: Maximum efficiency, single model capability          │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│            Different Models (Vocab Transfer)                    │
├────────────────────────────────────────────────────────────────┤
│ Memory: Nx models (e.g., 0.5B + 7B + 1.5B ≈ 18GB)             │
│ Transfer: Matrix multiplications per switch                     │
│   - Pseudoinverse: O(dim²·vocab) one-time                      │
│   - hidden @ W_out^+: O(batch·seq·dim·vocab)                   │
│   - vocab @ W_in: O(batch·seq·vocab·dim_new)                   │
│ Latency: ~10-100ms per transfer (depends on sequence length)   │
│ Use case: Flexibility, different model strengths               │
└────────────────────────────────────────────────────────────────┘
```

---

## Comparison with Alternatives

```
┌─────────────────┬──────────────┬──────────────────┬──────────────────┐
│ Method          │ KV Cache     │ Multi-Model      │ Realignment      │
├─────────────────┼──────────────┼──────────────────┼──────────────────┤
│ latent_mas      │ ✓ Efficient  │ ✗ Single model   │ ✓ Per-model      │
│                 │              │   only           │                  │
├─────────────────┼──────────────┼──────────────────┼──────────────────┤
│ embedding pass  │ ✗ No cache   │ ✓ Any models     │ ✗ Shared matrix  │
│                 │              │                  │   (incorrect!)   │
├─────────────────┼──────────────┼──────────────────┼──────────────────┤
│ HYBRID          │ ✓ When same  │ ✓ Vocab transfer │ ✓ Per-model      │
│ (this impl)     │   model      │   when different │   (correct!)     │
└─────────────────┴──────────────┴──────────────────┴──────────────────┘
```
