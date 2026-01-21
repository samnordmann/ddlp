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

    batch, seq, in_features, out_features = 2, 4, 16, 32
    if out_features % world_size != 0:
        raise RuntimeError("out_features must be divisible by world_size for this test")

    x = torch.randn(batch, seq, in_features, device=device)
    full_weight = torch.randn(out_features, in_features, device=device)
    full_bias = torch.randn(out_features, device=device)

    local_out = out_features // world_size
    start = rank * local_out
    end = start + local_out

    model = LinearColumnwise(
        in_features,
        out_features,
        bias=True,
        backend="pytorch",
        device=device,
    )
    with torch.no_grad():
        model.weight.copy_(full_weight[start:end])
        model.bias.copy_(full_bias[start:end])

    out = model(x)
    ref = torch.nn.functional.linear(x, full_weight, full_bias)

    max_diff = (out - ref).abs().max().item()
    if max_diff > 1e-5:
        raise AssertionError(f"LinearColumnwise mismatch: max_diff={max_diff}")

    dist.barrier()
    if rank == 0:
        print("LinearColumnwise CUDA test passed (max_diff=", max_diff, ")")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

