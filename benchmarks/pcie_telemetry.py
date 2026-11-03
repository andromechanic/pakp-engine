import torch

def main():
    num_elements = 256 * 1024 * 1024
    size_gb = (num_elements * 2) / (1024**3)

    cpu_tensor_normal = torch.empty(num_elements, dtype=torch.float16, pin_memory=False)
    cpu_tensor_pinned = torch.empty(num_elements, dtype=torch.float16, pin_memory=True)
    _ = torch.zeros(1024, device="cuda")

    def benchmark_transfer(tensor, non_blocking=False):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        _ = tensor.to("cuda", non_blocking=non_blocking)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        bandwidth_gbs = size_gb / (elapsed_ms / 1000.0)
        return elapsed_ms, bandwidth_gbs

    ms_normal, bw_normal = benchmark_transfer(cpu_tensor_normal, non_blocking=False)
    ms_pinned, bw_pinned = benchmark_transfer(cpu_tensor_pinned, non_blocking=True)

    print(f"Normal RAM -> VRAM : {ms_normal:.2f} ms | Speed: {bw_normal:.2f} GB/s")
    print(f"Pinned RAM -> VRAM : {ms_pinned:.2f} ms | Speed: {bw_pinned:.2f} GB/s")

if __name__ == "__main__":
    main()
