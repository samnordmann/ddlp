# DDLP: Distributed Deep Learning Primitives

DDLP is a high-performance, research-oriented library providing "speed-of-light" distributed primitives for deep learning on NVIDIA hardware. It focuses on maximizing communication/computation overlap using compiler-driven optimizations (via nvFuser) and hardware-specific features (Copy Engines, NVLink, NVLS).

## Mission

To enable deep exploration of the performance space for distributed primitives by treating communication and computation as a unified, fused pipeline.

## Features

- **Supported Primitives**:
  - `SP-TP-ColumnParallelLinear`: AllGather + Linear
- **Backend**: Primary backend leveraging **nvFuser** for host-initiated, zero-SM communication pipelines.

- **Target Primitives**:
  - `SP-TP-RowParallelLinear`: Linear + ReduceScatter
  - `TP-RowParallelLinear`: Linear + AllReduce
  - `MixtureOfExperts`: Dispatch, Routing, and Combine

## Installation

```bash
pip install -v -e ./python
```

## Structure

- `python/ddlp`: Python API and bindings.
- `cpp/`: C++ source implementations and headers.
- `tests/`: Unit tests.
- `benchmarks/`: Performance benchmarks (referencing DDLB).

## Dependencies

- PyTorch
- CUDA Toolkit
- MPI (Optional, for Communicator)
- nvFuser (Internal dependency)



