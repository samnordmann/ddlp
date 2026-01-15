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


def _get_dist_context(backend: str, process_group: Optional[torch.distributed.ProcessGroup]) -> _DistContext:
    if backend in ("pytorch", "fuser"):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "LinearColumnwise: torch.distributed must be initialized for pytorch/fuser backend"
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
        gathered_weight = [torch.empty_like(weight) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_weight, weight, group=self.process_group)
        full_weight = torch.cat(gathered_weight, dim=0)

        full_bias = None
        if bias is not None:
            gathered_bias = [torch.empty_like(bias) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_bias, bias, group=self.process_group)
            full_bias = torch.cat(gathered_bias, dim=0)

        return F.linear(input, full_weight, full_bias)


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
        backend: str = "cpp",
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.backend = backend

        self._dist = _get_dist_context(backend, process_group)
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

        if backend == "pytorch":
            self._backend = _PytorchBackend(self._dist.process_group)
        elif backend == "fuser":
            self._backend = _FuserBackend(self._dist.process_group)
        elif backend == "cpp":
            self._backend = _CppBackend(in_features, out_features)
        else:
            raise ValueError(f"LinearColumnwise: unsupported backend '{backend}'")

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._backend.forward(input, self.weight, self.bias, self._dist.world_size)

