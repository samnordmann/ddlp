# pylint: disable=too-many-instance-attributes
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
from torch import nn
from torch.nn import functional as F

import ddlp._C.primitives as _cpp_primitives
from ddlp.communicator import Communicator


@dataclass
class _DistContext:
    rank: int
    world_size: int
    process_group: Optional[torch.distributed.ProcessGroup]


def _backend_uses_torch_dist(backend: str) -> bool:
    return backend in ("pytorch", "fuser", "transformer_engine")


def _get_dist_context(backend: str, process_group: Optional[torch.distributed.ProcessGroup]) -> _DistContext:
    if _backend_uses_torch_dist(backend):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "LinearColumnwise: torch.distributed must be initialized for pytorch/fuser/transformer_engine backend"
            )
        rank = torch.distributed.get_rank(process_group)
        world_size = torch.distributed.get_world_size(process_group)
        return _DistContext(rank=rank, world_size=world_size, process_group=process_group)

    comm = Communicator()
    return _DistContext(rank=comm.rank(), world_size=comm.world_size(), process_group=None)


class _PytorchBackend:
    def __init__(self, process_group: Optional[torch.distributed.ProcessGroup]):
        self.process_group = process_group

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        world_size: int,
    ) -> torch.Tensor:
        local_out = F.linear(input, weight, bias)
        gathered_out = [torch.empty_like(local_out) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_out, local_out, group=self.process_group)
        return torch.cat(gathered_out, dim=-1)


class _FuserBackend(_PytorchBackend):
    def __init__(self, process_group: Optional[torch.distributed.ProcessGroup]):
        super().__init__(process_group)
        try:
            torch._C._jit_set_nvfuser_enabled(True)
        except Exception:
            pass

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        world_size: int,
    ) -> torch.Tensor:
        return super().forward(input, weight, bias, world_size)


class _CppBackend:
    def __init__(self, in_features: int, out_features: int):
        self.impl = _cpp_primitives.LinearColumnwiseImpl(in_features, out_features)

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        world_size: int,
    ) -> torch.Tensor:
        return self.impl.forward(input, weight, bias)


class _TransformerEngineBackend:
    def __init__(
        self,
        in_features: int,
        local_out_features: int,
        bias: bool,
        weight: nn.Parameter,
        bias_param: Optional[nn.Parameter],
        process_group: Optional[torch.distributed.ProcessGroup],
    ):
        try:
            import transformer_engine.pytorch as te
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "LinearColumnwise: transformer_engine backend requested but not available"
            ) from exc
        self.process_group = process_group
        self.linear = te.Linear(in_features, local_out_features, bias=bias)
        self.linear.weight = weight
        if bias:
            self.linear.bias = bias_param

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        world_size: int,
    ) -> torch.Tensor:
        local_out = self.linear(input)
        gathered_out = [torch.empty_like(local_out) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_out, local_out, group=self.process_group)
        return torch.cat(gathered_out, dim=-1)


class LinearColumnwise(nn.Module):
    """
    Tensor-parallel column-wise linear with all-gather of partial outputs.
    Mirrors torch.nn.Linear (global out_features) with sharded weight/bias per rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        backend: str = "fuser",
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if device is not None and device.type != "cuda":
            raise ValueError("LinearColumnwise: only CUDA devices are supported")
        self.in_features = in_features
        self.out_features = out_features
        self.backend = self._select_backend(backend)

        self._dist = _get_dist_context(self.backend, process_group)
        if out_features % self._dist.world_size != 0:
            raise ValueError("LinearColumnwise: out_features must be divisible by world_size")
        self.local_out_features = out_features // self._dist.world_size

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(self.local_out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.local_out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        if self.backend == "pytorch":
            self._backend = _PytorchBackend(self._dist.process_group)
        elif self.backend == "fuser":
            self._backend = _FuserBackend(self._dist.process_group)
        elif self.backend == "transformer_engine":
            self._backend = _TransformerEngineBackend(
                in_features,
                self.local_out_features,
                bias,
                self.weight,
                self.bias,
                self._dist.process_group,
            )
        elif self.backend == "cpp":
            self._backend = _CppBackend(in_features, out_features)
        else:
            raise ValueError(f"LinearColumnwise: unsupported backend '{self.backend}'")

    @staticmethod
    def _is_fuser_available() -> bool:
        if not torch.cuda.is_available():
            return False
        if not hasattr(torch._C, "_jit_set_nvfuser_enabled"):
            return False
        try:
            torch._C._jit_set_nvfuser_enabled(True)
        except Exception:
            return False
        return True

    @classmethod
    def _select_backend(cls, backend: str) -> str:
        if backend in ("auto", "fuser"):
            return "fuser" if cls._is_fuser_available() else "pytorch"
        return backend

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_cuda or not self.weight.is_cuda:
            raise RuntimeError("LinearColumnwise: only CUDA tensors are supported")
        if self.bias is not None and not self.bias.is_cuda:
            raise RuntimeError("LinearColumnwise: bias must be on CUDA")
        if _backend_uses_torch_dist(self.backend):
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError(
                    "LinearColumnwise: torch.distributed must be initialized for CUDA tensors"
                )
            world_size = torch.distributed.get_world_size(self._dist.process_group)
            if world_size != self._dist.world_size:
                raise RuntimeError(
                    "LinearColumnwise: torch.distributed world_size mismatch with module world_size"
                )
        else:
            world_size = self._dist.world_size
        return self._backend.forward(input, self.weight, self.bias, world_size)

