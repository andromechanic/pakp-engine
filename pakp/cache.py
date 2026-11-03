import torch
from typing import Optional, Tuple, Dict, Any, List
from transformers import DynamicCache


class PAKPAsymmetricCache(DynamicCache):
    """
    Predictive Asymmetric KV-Paging (PAKP) Cache for Hugging Face Transformers.
    Uses Centered Subspace + Min/Max Bounding-Box Descriptors for needle recall.
    """

    def __init__(
        self,
        num_layers: int,
        page_size: int = 64,
        top_k: int = 4,
        sink_size: int = 4,
        window_size: int = 256,
        bbox_weight: float = 1.0,
        mean_weight: float = 0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.page_size = page_size
        self.top_k = top_k
        self.sink_size = sink_size
        self.window_size = window_size
        self.bbox_weight = bbox_weight
        self.mean_weight = mean_weight

        self.vram_sinks_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self.vram_sinks_v: List[Optional[torch.Tensor]] = [None] * num_layers
        self.vram_window_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self.vram_window_v: List[Optional[torch.Tensor]] = [None] * num_layers

        self.vram_global_k_mean: List[Optional[torch.Tensor]] = [None] * num_layers
        self.vram_page_mean: List[Optional[torch.Tensor]] = [None] * num_layers
        self.vram_page_min: List[Optional[torch.Tensor]] = [None] * num_layers
        self.vram_page_max: List[Optional[torch.Tensor]] = [None] * num_layers

        self.host_pinned_pages: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]

        self.prefetch_stream = torch.cuda.Stream(device="cuda")
        self.prefetch_events = [torch.cuda.Event() for _ in range(num_layers)]
        self.prefetched_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self.prefetched_v: List[Optional[torch.Tensor]] = [None] * num_layers

        self.is_decoding: bool = False
        self.scoring_mode: str = "pakp_bbox"
        self.total_lookahead_prefetches: int = 0
        self.last_selected_pages: List[int] = []

    @torch.no_grad()
    def evict_prefill_to_pinned_host(self) -> None:
        for l_idx in range(self.num_layers):
            full_k = self.vram_window_k[l_idx]
            full_v = self.vram_window_v[l_idx]
            if full_k is None:
                continue

            total_len = full_k.size(2)
            self.vram_sinks_k[l_idx] = full_k[:, :, :self.sink_size, :].contiguous()
            self.vram_sinks_v[l_idx] = full_v[:, :, :self.sink_size, :].contiguous()

            evictable_tokens = total_len - self.sink_size - self.window_size
            num_pages = max(0, evictable_tokens // self.page_size)
            evict_end = self.sink_size + num_pages * self.page_size

            middle_k = full_k[:, :, self.sink_size:evict_end, :].contiguous()
            middle_v = full_v[:, :, self.sink_size:evict_end, :].contiguous()

            self.vram_window_k[l_idx] = full_k[:, :, evict_end:, :].contiguous()
            self.vram_window_v[l_idx] = full_v[:, :, evict_end:, :].contiguous()

            if num_pages > 0:
                k_mean_global = middle_k[0].mean(dim=1, keepdim=True)
                self.vram_global_k_mean[l_idx] = k_mean_global

                k_centered = (
                    (middle_k[0] - k_mean_global)
                    .view(middle_k.size(1), num_pages, self.page_size, middle_k.size(3))
                    .permute(1, 0, 2, 3)
                    .contiguous()
                )

                self.vram_page_mean[l_idx] = k_centered.mean(dim=2)
                self.vram_page_min[l_idx] = k_centered.amin(dim=2)
                self.vram_page_max[l_idx] = k_centered.amax(dim=2)

                k_pages = middle_k.view(1, middle_k.size(1), num_pages, self.page_size, middle_k.size(3))
                v_pages = middle_v.view(1, middle_v.size(1), num_pages, self.page_size, middle_v.size(3))

                for p in range(num_pages):
                    kp = k_pages[:, :, p, :, :].contiguous()
                    vp = v_pages[:, :, p, :, :].contiguous()
                    pinned_page = torch.empty((2, *kp.shape), dtype=kp.dtype, pin_memory=True)
                    pinned_page[0].copy_(kp, non_blocking=True)
                    pinned_page[1].copy_(vp, non_blocking=True)
                    self.host_pinned_pages[l_idx].append(pinned_page)

            del full_k, full_v, middle_k, middle_v

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def trigger_lookahead_prefetch(self, layer_idx: int, q_rope_grouped: torch.Tensor) -> None:
        num_pages = len(self.host_pinned_pages[layer_idx])
        if num_pages == 0 or self.top_k == 0 or self.scoring_mode == "sliding_window":
            self.prefetched_k[layer_idx] = None
            self.prefetched_v[layer_idx] = None
            return

        k_fetch = min(self.top_k, num_pages)
        c_mean = self.vram_page_mean[layer_idx]

        if self.scoring_mode == "naive_mean":
            c_uncentered = c_mean + self.vram_global_k_mean[layer_idx].permute(1, 0, 2)
            scores = torch.einsum("hgd,phd->p", q_rope_grouped, c_uncentered)
        else:
            c_min = self.vram_page_min[layer_idx]
            c_max = self.vram_page_max[layer_idx]

            prod_max = torch.einsum("hgd,phd->phg", q_rope_grouped, c_max)
            prod_min = torch.einsum("hgd,phd->phg", q_rope_grouped, c_min)
            bbox_score = torch.maximum(prod_max, prod_min).sum(dim=-1).amax(dim=-1)
            mean_score = torch.einsum("hgd,phd->phg", q_rope_grouped, c_mean).mean(dim=(1, 2))
            scores = (self.bbox_weight * bbox_score) + (self.mean_weight * mean_score)

        _, top_idx = torch.topk(scores, k=k_fetch, dim=0)
        top_indices = sorted(top_idx.cpu().tolist())

        if layer_idx == 16:
            self.last_selected_pages = top_indices

        with torch.cuda.stream(self.prefetch_stream):
            k_list, v_list = [], []
            for idx in top_indices:
                page = self.host_pinned_pages[layer_idx][idx]
                k_list.append(page[0].to("cuda", non_blocking=True))
                v_list.append(page[1].to("cuda", non_blocking=True))
            self.prefetched_k[layer_idx] = torch.cat(k_list, dim=2)
            self.prefetched_v[layer_idx] = torch.cat(v_list, dim=2)
            self.prefetch_events[layer_idx].record(self.prefetch_stream)

        self.total_lookahead_prefetches += 1

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_decoding:
            if self.vram_window_k[layer_idx] is None:
                self.vram_window_k[layer_idx] = key_states.contiguous()
                self.vram_window_v[layer_idx] = value_states.contiguous()
            else:
                self.vram_window_k[layer_idx] = torch.cat([self.vram_window_k[layer_idx], key_states], dim=2)
                self.vram_window_v[layer_idx] = torch.cat([self.vram_window_v[layer_idx], value_states], dim=2)
            return self.vram_window_k[layer_idx], self.vram_window_v[layer_idx]

        self.vram_window_k[layer_idx] = torch.cat([self.vram_window_k[layer_idx], key_states], dim=2)
        self.vram_window_v[layer_idx] = torch.cat([self.vram_window_v[layer_idx], value_states], dim=2)

        parts_k = [self.vram_sinks_k[layer_idx]]
        parts_v = [self.vram_sinks_v[layer_idx]]

        if self.prefetched_k[layer_idx] is not None and self.prefetched_k[layer_idx].size(2) > 0:
            torch.cuda.current_stream().wait_event(self.prefetch_events[layer_idx])
            parts_k.append(self.prefetched_k[layer_idx])
            parts_v.append(self.prefetched_v[layer_idx])

        parts_k.append(self.vram_window_k[layer_idx])
        parts_v.append(self.vram_window_v[layer_idx])

        return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self.is_decoding:
            return 0 if self.vram_window_k[layer_idx] is None else self.vram_window_k[layer_idx].size(2)
        return (
            self.vram_sinks_k[layer_idx].size(2)
            + (0 if self.prefetched_k[layer_idx] is None else self.prefetched_k[layer_idx].size(2))
            + self.vram_window_k[layer_idx].size(2)
        )
