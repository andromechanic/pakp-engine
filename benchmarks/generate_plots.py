from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 300
})

def plot_latency_breakdown():
    methods = [
        "Naive Full Offload\n(32k over PCIe)",
        "Sync Top-4 Paging\n(No Lookahead)",
        "PAKP Dual-Stream\n(Ours - Overlapped)",
        "Ideal In-VRAM\n(Zero PCIe Bound)"
    ]
    latencies = [321.26, 31.60, 25.54, 24.90]
    colors = ["#d9534f", "#f0ad4e", "#2b5c8f", "#5cb85c"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(methods, latencies, color=colors, width=0.52, edgecolor="black", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Per-Token Latency (ms, log scale)")
    ax.set_title("32,768-Token Decoding Latency on RTX 4050 (6GB VRAM + PCIe Gen4)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, lat in zip(bars, latencies):
        tok_s = 1000.0 / lat
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            lat * 1.15,
            f"{lat:.2f} ms\n({tok_s:.1f} tok/s)",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold"
        )

    ax.set_ylim(10, 800)
    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "latency_breakdown.png")
    fig.savefig(ASSETS_DIR / "latency_breakdown.pdf")
    plt.close(fig)

def plot_vram_scaling():
    ctx_k = np.array([2, 4, 8, 16, 32, 64])
    standard_vram = 2.95 + (ctx_k * 0.109)
    pakp_vram = 3.55 + (ctx_k * 0.0025)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ctx_k, standard_vram, marker="o", linewidth=2.4, color="#d9534f", label="Standard HuggingFace KV-Cache (VRAM)")
    ax.plot(ctx_k, pakp_vram, marker="s", linewidth=2.4, color="#2b5c8f", label="PAKP Asymmetric Paging (Ours)")
    ax.axhline(6.0, color="black", linestyle="--", linewidth=1.5, label="NVIDIA RTX 4050 Hardware Limit (6.0 GB VRAM)")

    ax.fill_between(ctx_k, 6.0, 10.5, color="#d9534f", alpha=0.12, label="CUDA Out-Of-Memory (OOM) Zone")
    ax.set_xlabel("Context Length (Thousands of Tokens)")
    ax.set_ylabel("Allocated GPU VRAM (GB)")
    ax.set_title("VRAM Scaling Across Context Lengths (RTX 4050 6GB + 32GB Host RAM)")
    ax.set_xticks(ctx_k)
    ax.set_xticklabels([f"{k}K" for k in ctx_k])
    ax.set_ylim(2.0, 10.2)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9.5)

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "vram_scaling.png")
    fig.savefig(ASSETS_DIR / "vram_scaling.pdf")
    plt.close(fig)

if __name__ == "__main__":
    plot_latency_breakdown()
    plot_vram_scaling()
    print(f"[Plots Generated] Saved PNG & PDF figures to {ASSETS_DIR}")
