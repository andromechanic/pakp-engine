# PAKP: Predictive Asymmetric KV-Paging for Edge GPUs

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)]([https://pytorch.org/](https://pytorch.org/))
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**PAKP (`pakp-engine`)** is a two-tier asymmetric KV-cache inference engine and research framework that breaks the **6 GB VRAM wall** on consumer GPUs (e.g., **NVIDIA RTX 4050 Laptop GPU**). By pairing **32 GB Page-Locked Host RAM (`cudaMallocHost`)** with **Anisotropy-Corrected Bounding-Box Lookahead** and **Dual-Stream Asynchronous DMA Prefetching**, PAKP hides **90.5% of PCIe Gen4 latency** behind GPU compute while retaining **100% passkey recall**.

---

## Visual Benchmarks (ASUS TUF F15 — RTX 4050 6GB + 32GB Host RAM)

<p align="center">
  <img src="assets/latency_breakdown.png" width="48%" alt="Latency Breakdown" />
  <img src="assets/vram_scaling.png" width="48%" alt="VRAM Scaling" />
</p>

---

## Core Technical Innovations

1. **Two-Tier Asymmetric Memory Topology:** Retains only Attention Sinks ($S_{\text{sink}}=4$), a sliding Local Window ($S_{\text{win}}=256$), and a compact Subspace Index inside GPU VRAM, offloading historical 64-token KV pages into 32 GB Pinned Host RAM.
2. **Centered Subspace Bounding-Box Index:** Removes key-space anisotropy by subtracting the per-layer global Key mean $\bar{K}_L$ and storing channel-wise extrema $(K_{\min}^{(p)}, K_{\max}^{(p)})$ alongside page centroids $\mu_p$, preventing single-token passkeys from being diluted across 64-token pages.
3. **RoPE-Aligned Cross-Layer Lookahead:** Exploits residual stream continuity ($\cos(h_L, h_{L+1}) = 0.9842$) to project $\hat{Q}_{L+1}^{\text{RoPE}}$ at Layer $L$, overlapping PCIe Gen4 DMA transfers (`prefetch_stream`) with Layer $L$ GPU compute (`compute_stream`).

---

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pakp import enable_pakp

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="cuda")

# Attach PAKP Two-Tier Asymmetric Paging in one line
pakp_cache, hooks = enable_pakp(model, page_size=64, top_k=4, sink_size=4, window_size=256)
```

---

## Reproducing the Paper Benchmarks

Run the complete empirical suite on your local CUDA GPU:

```bash
# 1. Measure Host-to-Device PCIe Gen4 Bandwidth (Pageable vs Pinned DMA)
python benchmarks/pcie_telemetry.py

# 2. Run 28-Layer Dual-Stream DMA Overlap Benchmark (32,768 tokens)
python benchmarks/dma_overlap_bench.py

# 3. Run Live HuggingFace LLM Passkey Ablation Suite (Qwen2.5-1.5B-Instruct)
python benchmarks/live_llm_ablation.py

# 4. Generate High-Resolution Figures (PNG & PDF for LaTeX Paper)
python benchmarks/generate_plots.py
```

---

## Empirical Telemetry Summary

### Table 1: 32,768-Token Dual-Stream DMA Overlap (RTX 4050 6GB)
| Execution Strategy | Per-Token Latency | Throughput | PCIe Overhead Hidden |
| :--- | :---: | :---: | :---: |
| Naive Full KV Offload (3.50 GB over PCIe) | `321.26 ms` | `3.1 tok/s` | `0.0%` |
| Synchronous Top-4 Paging (No Lookahead) | `31.60 ms` | `31.6 tok/s` | `0.0%` |
| **PAKP Dual-Stream Lookahead (Ours)** | **`25.54 ms`** | **`39.2 tok/s`** | **`90.5%`** |
| *Ideal In-VRAM Upper Bound (Zero PCIe)* | *`24.90 ms`* | *`40.2 tok/s`* | *—* |

### Table 2: Live LLM Passkey Retrieval Ablation (`Qwen2.5-1.5B-Instruct`)
| Method | Evicted Page Hit | Active VRAM | Speed | Generated Output |
| :--- | :---: | :---: | :---: | :--- |
| 1. Sliding Window (`Sinks + Win`) | `False` | `3.61 GB` | `46.6 tok/s` | `"123456"` *(Amnesia)* |
| 2. Uncentered Mean Paging | `False` | `3.64 GB` | `26.8 tok/s` | *Misses needle page* |
| **3. PAKP Centered BBox (Ours)** | **`True`** | **`3.64 GB`** | **`27.1 tok/s`** | **`"7392-OMEGA."` (100% Recall)** |

---

## Academic Paper
The full LaTeX source for the accompanying research paper is located in [`PAKP_research_paper.pdf`](PAKP_research_paper.pdf).

```bibtex
@article{akhil2026pakp,
  title={PAKP: Predictive Asymmetric KV-Paging via Centered Bounding-Box Lookahead for Sub-6GB Edge GPUs},
  author={Akhil M},
  year={2026}
}
```

## License
Released under the [MIT License](LICENSE).
