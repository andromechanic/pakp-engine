import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pakp import PAKPAsymmetricCache, attach_pakp_lookahead_hooks

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PAGE_SIZE = 64
TOP_K_PAGES = 4
SINK_TOKENS = 4
LOCAL_WINDOW = 256


def test_mode(model, tokenizer, input_ids, expected_page, mode="pakp_bbox"):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    seq_len = input_ids.size(1)
    pakp_cache = PAKPAsymmetricCache(
        num_layers=model.config.num_hidden_layers,
        page_size=PAGE_SIZE,
        top_k=TOP_K_PAGES if mode != "sliding_window" else 0,
        sink_size=SINK_TOKENS,
        window_size=LOCAL_WINDOW,
    )
    pakp_cache.scoring_mode = mode

    hooks = attach_pakp_lookahead_hooks(model, pakp_cache) if mode != "sliding_window" else []

    with torch.no_grad():
        # Backbone-only prefill (avoids allocating 1.08 GB of unused lm_head logits)
        pakp_cache.is_decoding = False
        _ = model.model(input_ids=input_ids[:, :-1], past_key_values=pakp_cache, use_cache=True)
        pakp_cache.evict_prefill_to_pinned_host()
        torch.cuda.reset_peak_memory_stats()

        pakp_cache.is_decoding = True
        curr_token = input_ids[:, -1:]
        curr_pos = seq_len - 1
        generated_ids = []
        t0 = time.perf_counter()
        for _ in range(10):
            pos_ids = torch.tensor([[curr_pos]], device="cuda")
            out = model(input_ids=curr_token, position_ids=pos_ids, past_key_values=pakp_cache, use_cache=True)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token.item())
            curr_token = next_token
            curr_pos += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    for h in hooks:
        h.remove()

    ans = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    p_hit = expected_page in pakp_cache.last_selected_pages if mode != "sliding_window" else False
    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    selected = list(pakp_cache.last_selected_pages)

    del pakp_cache
    gc.collect()
    torch.cuda.empty_cache()
    return ans, p_hit, selected, vram_gb, len(generated_ids) / dt


def run_ablation():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda")
    model.eval()

    needle = "The secret launch authorization code for Project Titan is 7392-OMEGA. "
    filler = "Distributed systems rely on consensus protocols, asynchronous remote procedure calls, and deterministic state replication across fault-tolerant storage nodes. "
    user_content = (
        "Read the following system log archive carefully and answer the question at the end.\n\n"
        + filler * 15
        + needle
        + filler * 130
        + "\nQuestion: What is the exact secret launch authorization code for Project Titan?"
    )
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "The secret launch authorization code for Project Titan is"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    needle_ids = tokenizer("7392-OMEGA", add_special_tokens=False).input_ids
    needle_pos = next(i for i in range(input_ids.size(1)) if input_ids[0, i:i+len(needle_ids)].tolist() == needle_ids)
    expected_page = (needle_pos - SINK_TOKENS) // PAGE_SIZE

    print(f"\n[Ablation Study] Context: {input_ids.size(1):,} tokens | Target Needle Page: #{expected_page}")
    print("=" * 98)
    print(f"{'Method':<29} | {'Selected Pages':<18} | {'Page Hit':<8} | {'VRAM':<7} | {'Speed':<8} | {'Output'}")
    print("-" * 98)

    results = {}
    for mode, label in [
        ("sliding_window", "1. Sliding Window (Sinks+Win)"),
        ("naive_mean",     "2. Uncentered Mean Paging"),
        ("pakp_bbox",      "3. PAKP Centered BBox (Ours)")
    ]:
        ans, p_hit, pages, vram, spd = test_mode(model, tokenizer, input_ids, expected_page, mode)
        print(f"{label:<29} | {str(pages):<18} | {str(p_hit):<8} | {vram:.2f} GB | {spd:5.1f} t/s | {ans[:22]}")
        results[mode] = {
            "selected_pages": pages,
            "page_hit": p_hit,
            "vram_gb": round(vram, 2),
            "tokens_per_sec": round(spd, 1),
            "output": ans
        }
    print("=" * 98)

    out_path = Path(__file__).resolve().parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_ablation()
