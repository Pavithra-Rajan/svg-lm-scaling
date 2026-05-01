# svg-lm-scaling

**Scaling Laws for SVG Language Models**: A study of how decoder-only Transformer language models scale when trained exclusively on Scalable Vector Graphics. Covers data cleaning, BPE tokenization, five GPT-style architectures from $853$K to $85.3$M parameters, learning-rate sweeps, scaling-law fitting under both Standard Parameterization (SP) and Maximal Update Parameterization (mup), and a qualitative evaluation of generated SVG samples.

The full write-up with the approach, analysis, tables, and figures lives in [`Report.pdf`](./Report.pdf). This README is a summary.

> **TLDR**
> - Best validation loss: **`sp_medium` at $0.6322$ nats/token** (10.8M params, 1 epoch over 145.6M tokens).
> - SP scaling at a fixed Tiny-tuned LR is **non-monotonic** as Large and XL diverge.
> - mup with the same LR ($6{\times}10^{-3}$) is **stable at all five scales**, validating LR-transfer.
> - Generated samples are $98\%$ XML-valid, $98\%$ renderable, $100\%$ use a `viewBox`.


## Model Architecture

![architecture](drive_directories/results/model_architecture.png)

A nanoGPT-style decoder with pre-LayerNorm, learned absolute positional embeddings, and 4&times;-expansion GELU MLP blocks. The same template is reused at every scale; only $d_{\text{model}}$, depth, and head count vary.

| Model  | $d_{\text{model}}$ | Layers | Heads | $d_h$ | Params (non-emb.) |
|--------|-------:|-------:|------:|------:|------------------:|
| Tiny   | 128 | 4  | 4  | 32 | 853,120        |
| Small  | 192 | 6  | 6  | 32 | 2,755,008   |
| Medium | 384 | 6  | 6  | 64 | 10,818,432  |
| Large  | 512 | 10 | 8  | 64 | 31,730,176  |
| XL     | 768 | 12 | 12 | 64 |  85,347,072        |

All scales share `vocab_size=4096`, `block_size=512`, and `dropout=0`. Token and output embeddings are tied (SP) or untied via `MuReadout` (mup).
