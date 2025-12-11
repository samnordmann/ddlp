import torch
import time
from ddlp.primitives import SPTPRowParallelLinear

def benchmark_linear():
    print("Benchmarking SP-TP-RowParallelLinear...")
    B, M, K, N = 32, 128, 1024, 1024
    model = SPTPRowParallelLinear(K, N).cuda()
    x = torch.randn(B, M, K).cuda()
    
    # Warmup
    for _ in range(5):
        _ = model(x)
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model(x)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"Time per iter: {(end - start)/100 * 1000:.2f} ms")

if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark_linear()
    else:
        print("CUDA not available, skipping benchmark")



