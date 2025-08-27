# DeepSeek — Latent Attention + RoPE
An advanced, compute-efficient attention module inspired by *Attention Is All You Need* and Perceiver-like latent mechanisms.
This repository provides:
- A clear explanation of the idea
- A compact PyTorch implementation of a **latent attention** block (a single shared latent attention that provides keys/values)
- RoPE (rotary positional embeddings) applied to Q/K
- Example usage and integration into transformer blocks

---

## Idea / Motivation

Standard multi-head self-attention computes Q, K, V per token and performs O(N²) attention for length N.  
**DeepSeek** reduces this by introducing a small set of learnable *latent tokens* which:
1. Aggregate information from the full token set (forming compact K/V representation).
2. Are then attended to by queries from the sequence.

Rough analogy: imagine multiple doors (queries). Instead of creating a unique key for each door, we have a single set of master keys (latents). Each query uses the same set of master keys to open the door. This reduces compute/memory while still preserving expressive cross-token interactions.

Two important features:
- **Single shared latent attention** (one latent bank used across heads or with headwise projection).
- **RoPE (Rotary Positional Embeddings)** replaces absolute positional embeddings, encoding relative position by rotating Q/K vectors.

---

## Architecture (high level)

Input X (N, D)  ──> Linear → Q (N, Dq)
                 ──> Linear → K_input (N, Dk)
                 ──> Linear → V_input (N, Dv)

Learnable Latents L (M, Dlatent)

Step A: Latents attend to input to form latent K/V:
  L' = Attn_cross(L, K_input, V_input)   # latents absorb input info

Step B: Sequence queries attend to latents:
  Out = Attn(Q, K_latent, V_latent)

Optional: feed-forward block & residuals like a standard transformer.

---

## Benefits
- Lower compute for long sequences: attention scales O(N*M) instead of O(N²) (M ≪ N)
- Flexible — use more latents for higher capacity, fewer latents for lower compute
- RoPE preserves relative position information efficiently

---

## Files
- `deepseek/latent_attention.py` — core module (PyTorch)
- `examples/quick_demo.py` — usage demo
- `README.md` — this document

---

## Usage (quick)

```py
from deepseek.latent_attention import LatentAttentionBlock

block = LatentAttentionBlock(
    dim=512,           # embedding dimension
    num_heads=8,
    num_latents=64,    # much smaller than seq length
    latent_dim=512,
    rope=True
)

x = torch.randn(16, 1024, 512)  # batch, seq_len, dim
out = block(x)                  # same shape as x
