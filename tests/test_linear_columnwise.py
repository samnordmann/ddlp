import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist

repo_root = Path(__file__).resolve().parents[1]
python_root = repo_root / "python"
sys.path.insert(0, str(python_root))

from ddlp.primitives import LinearColumnwise


def _init_dist():
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)),
        )
    )
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=local_rank,
    )
    return rank, world_size, local_rank


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")
    torch.manual_seed(1234)

    rank, world_size, local_rank = _init_dist()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    m, k, n = 32, 16, 24
    if m % world_size != 0:
        raise RuntimeError("m must be divisible by world_size for this test")
    m_local = m // world_size

    full_input = torch.randn(m, k, device=device)
    weight = torch.randn(k, n, device=device)
    bias = torch.randn(n, device=device)
    input_shard = full_input[rank * m_local : (rank + 1) * m_local]

    ref = torch.matmul(full_input, weight) + bias
    backends = ["pytorch", "fuser"]
    for backend in backends:
        model = LinearColumnwise(
            in_features=k,
            out_features=n,
            bias=True,
            backend=backend,
            device=device,
        )
        with torch.no_grad():
            model.weight.copy_(weight)
            model.bias.copy_(bias)
        output = model(input_shard)
        max_diff = (output - ref).abs().max().item()
        tol = 1e-2
        if max_diff > tol:
            raise AssertionError(
                f"LinearColumnwise {backend} mismatch: max_diff={max_diff}"
            )
    dist.barrier()
    if rank == 0:
        print("LinearColumnwise CUDA test passed (backends=", ",".join(backends), ")")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
