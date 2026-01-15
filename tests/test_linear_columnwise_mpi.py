import torch

from ddlp.communicator import Communicator
from ddlp.primitives import LinearColumnwise


def main():
    comm = Communicator()
    rank = comm.rank()
    world_size = comm.world_size()

    torch.manual_seed(1234)

    batch, seq, in_features, out_features = 2, 4, 16, 32
    if out_features % world_size != 0:
        raise RuntimeError("out_features must be divisible by world_size for this test")

    x = torch.randn(batch, seq, in_features)
    full_weight = torch.randn(out_features, in_features)
    full_bias = torch.randn(out_features)

    local_out = out_features // world_size
    start = rank * local_out
    end = start + local_out

    model = LinearColumnwise(in_features, out_features, bias=True, backend="cpp")
    with torch.no_grad():
        model.weight.copy_(full_weight[start:end])
        model.bias.copy_(full_bias[start:end])

    out = model(x)
    ref = torch.nn.functional.linear(x, full_weight, full_bias)

    max_diff = (out - ref).abs().max().item()
    if max_diff > 1e-5:
        raise AssertionError(f"LinearColumnwise mismatch: max_diff={max_diff}")

    comm.barrier()
    if rank == 0:
        print("LinearColumnwise MPI test passed (max_diff=", max_diff, ")")
    comm.finalize()


if __name__ == "__main__":
    main()

