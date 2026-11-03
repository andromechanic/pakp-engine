import torch
from typing import List, Tuple
from .cache import PAKPAsymmetricCache


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1) if cos.dim() == 3 else cos
    sin = sin.unsqueeze(1) if sin.dim() == 3 else sin
    return (q * cos) + (_rotate_half(q) * sin)


def attach_pakp_lookahead_hooks(model: torch.nn.Module, pakp_cache: PAKPAsymmetricCache) -> List:
    layers = model.model.layers
    rotary_emb = model.model.rotary_emb
    num_layers = len(layers)
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_q_heads
    group_size = num_q_heads // num_kv_heads

    def make_hook(l_idx: int):
        def pre_hook(module, args, kwargs):
            if not pakp_cache.is_decoding or pakp_cache.scoring_mode == "sliding_window":
                return
            hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            if hidden_states is None or hidden_states.size(1) != 1:
                return

            position_embeddings = kwargs.get("position_embeddings", None)
            position_ids = kwargs.get("position_ids", None)

            target_layers = [0, 1] if l_idx == 0 else ([l_idx + 1] if l_idx + 1 < num_layers else [])
            for t_idx in target_layers:
                if len(pakp_cache.host_pinned_pages[t_idx]) == 0:
                    continue
                next_layer = layers[t_idx]
                normed_h = next_layer.input_layernorm(hidden_states)
                q = next_layer.self_attn.q_proj(normed_h)
                q = q.view(1, 1, num_q_heads, head_dim).transpose(1, 2)

                if position_embeddings is not None:
                    cos, sin = position_embeddings
                else:
                    cos, sin = rotary_emb(q, position_ids)
                q_rope = _apply_rope(q, cos, sin)

                q_grouped = q_rope[0, :, 0, :].view(num_kv_heads, group_size, head_dim)
                pakp_cache.trigger_lookahead_prefetch(t_idx, q_grouped)

        return pre_hook

    handles = []
    for i in range(num_layers):
        handles.append(layers[i].register_forward_pre_hook(make_hook(i), with_kwargs=True))
    return handles


def enable_pakp(
    model: torch.nn.Module,
    page_size: int = 64,
    top_k: int = 4,
    sink_size: int = 4,
    window_size: int = 256,
) -> Tuple[PAKPAsymmetricCache, List]:
    cache = PAKPAsymmetricCache(
        num_layers=model.config.num_hidden_layers,
        page_size=page_size,
        top_k=top_k,
        sink_size=sink_size,
        window_size=window_size,
    )
    hooks = attach_pakp_lookahead_hooks(model, cache)
    return cache, hooks
