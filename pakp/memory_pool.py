import math
import torch


class PAKPMemoryEngine:
    """
    Predictive Asymmetric KV-Paging (PAKP) Core Memory Engine.
    Manages Tier-1 (VRAM Sinks, Local Window, Centroid Index, Ping-Pong Buffers)
    and Tier-2 (Pinned Host RAM Episodic Page Pool) across Dual CUDA Streams.
    """
    def __init__(
        self,
        num_layers: int = 28,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        max_pages: int = 512,
        page_size: int = 64,
        top_k_pages: int = 4,
        sink_tokens: int = 4,
        local_window: int = 512,
        dtype: torch.dtype = torch.float16
    ):
        self.L = num_layers
        self.H_kv = num_kv_heads
        self.D = head_dim
        self.max_pages = max_pages
        self.page_size = page_size
        self.top_k = top_k_pages
        self.dtype = dtype

        self.host_kv_pool = [
            torch.randn(
                (2, max_pages, page_size, num_kv_heads, head_dim),
                dtype=dtype,
                pin_memory=True
            )
            for _ in range(num_layers)
        ]

        self.vram_centroids = torch.randn(
            (num_layers, max_pages, num_kv_heads, head_dim),
            dtype=dtype,
            device="cuda"
        )

        self.active_static_tokens = sink_tokens + local_window
        self.vram_static_kv = torch.randn(
            (num_layers, 2, self.active_static_tokens, num_kv_heads, head_dim),
            dtype=dtype,
            device="cuda"
        )

        self.vram_staging_kv = torch.empty(
            (2, 2, top_k_pages, page_size, num_kv_heads, head_dim),
            dtype=dtype,
            device="cuda"
        )

        self.compute_stream = torch.cuda.default_stream()
        self.prefetch_stream = torch.cuda.Stream(device="cuda")
        self.prefetch_events = [torch.cuda.Event() for _ in range(num_layers)]

    @torch.no_grad()
    def select_top_k_pages(self, query_lookahead: torch.Tensor, layer_idx: int) -> torch.Tensor:
        centroids = self.vram_centroids[layer_idx]
        scores = torch.einsum("hd,mhd->m", query_lookahead, centroids)
        _, top_indices = torch.topk(scores, k=self.top_k, dim=0)
        return top_indices.to("cpu", non_blocking=False)

    @torch.no_grad()
    def async_prefetch_layer(self, layer_idx: int, slot: int, page_indices: torch.Tensor):
        with torch.cuda.stream(self.prefetch_stream):
            selected_host_pages = self.host_kv_pool[layer_idx][:, page_indices]
            self.vram_staging_kv[slot].copy_(selected_host_pages, non_blocking=True)
            self.prefetch_events[layer_idx].record(self.prefetch_stream)


def simulate_layer_compute(q, k_static, v_static, k_paged, v_paged, mlp_w1, mlp_w2, hidden):
    k_needles = k_paged.view(-1, k_static.size(1), k_static.size(2))
    v_needles = v_paged.view(-1, v_static.size(1), v_static.size(2))

    k_active = torch.cat([k_static, k_needles], dim=0)
    v_active = torch.cat([v_static, v_needles], dim=0)

    k_gqa = k_active.repeat_interleave(3, dim=1)
    v_gqa = v_active.repeat_interleave(3, dim=1)

    scores = torch.einsum("hd,shd->hs", q, k_gqa) / math.sqrt(128)
    probs = torch.softmax(scores, dim=-1)
    attn_out = torch.einsum("hs,shd->hd", probs, v_gqa).reshape(1, -1)

    ffn_mid = torch.nn.functional.silu(hidden @ mlp_w1)
    out = ffn_mid @ mlp_w2 + attn_out
    return out
