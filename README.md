# DDLP: Distributed Deep Learning Primitives

DDLP is a lightweight, pure-Python library providing fused communication/computation primitives for distributed deep learning on NVIDIA GPUs.

## Primitive

- **`LinearColumnwise`**: Tensor-parallel column-wise linear layer (local GEMM + AllGather). Supports three backends:
  - `pytorch` -- pure PyTorch (`F.linear` + `torch.distributed.all_gather`)
  - `fuser` -- nvFuser-accelerated (requires `nvfuser_direct`)
  - `transformer_engine` -- Transformer Engine integration (requires `transformer_engine`)

## Installation

```bash
pip install -e ./python
```

Optional backends:

```bash
pip install -e "./python[fuser]"        # nvFuser backend
pip install -e "./python[te]"           # Transformer Engine backend
```

## Usage

```python
import torch
import torch.distributed as dist
from ddlp.primitives import LinearColumnwise

dist.init_process_group(backend="nccl")
model = LinearColumnwise(in_features=1024, out_features=4096, backend="pytorch", device="cuda")
output = model(torch.randn(2, 128, 1024, device="cuda"))
```

## Testing

```bash
torchrun --nproc_per_node=<N> tests/test_linear_columnwise.py
```

## Dependencies

- **Required:** PyTorch (with CUDA and `torch.distributed`)
- **Optional:** `nvfuser_direct` (fuser backend), `transformer_engine` (TE backend)
