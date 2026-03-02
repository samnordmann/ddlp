# pylint: disable=too-many-instance-attributes
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math
import os
import torch
from torch import nn

from .heuristics import Heuristic, DEFAULT_HEURISTIC

_DDLP_DEBUG = os.environ.get("DDLP_DEBUG", "0") == "1"


@dataclass
class _DistContext:
    rank: int
    world_size: int
    process_group: Optional[torch.distributed.ProcessGroup]


def _backend_uses_torch_dist(backend: str) -> bool:
    return backend in ("auto", "pytorch", "fuser")


def _get_dist_context(backend: str, process_group: Optional[torch.distributed.ProcessGroup]) -> _DistContext:
    if _backend_uses_torch_dist(backend):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "LinearColumnwise: torch.distributed must be initialized"
            )
        rank = torch.distributed.get_rank(process_group)
        world_size = torch.distributed.get_world_size(process_group)
        return _DistContext(rank=rank, world_size=world_size, process_group=process_group)

    raise ValueError(f"LinearColumnwise: unsupported backend '{backend}'")


class _PytorchBackend:
    def __init__(self, process_group: Optional[torch.distributed.ProcessGroup]):
        self.process_group = process_group
        self._gathered: Optional[torch.Tensor] = None

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        world_size: int,
    ) -> torch.Tensor:
        input_2d = input.reshape(-1, input.shape[-1])
        if self._gathered is None or self._gathered.shape[0] != input_2d.shape[0] * world_size:
            self._gathered = torch.empty(
                input_2d.shape[0] * world_size, input_2d.shape[1],
                dtype=input.dtype, device=input.device,
            )
        torch.distributed.all_gather_into_tensor(
            self._gathered, input_2d, group=self.process_group,
        )
        if bias is not None:
            return torch.addmm(bias, self._gathered, weight)
        return torch.mm(self._gathered, weight)


class _FuserBackend:
    """nvFuser multidevice backend: all-gather + matmul fused in a single executor."""

    def __init__(
        self,
        process_group: Optional[torch.distributed.ProcessGroup],
        world_size: int,
    ):
        self.process_group = process_group
        self._world_size = world_size
        try:
            import nvfuser_direct as nvfuser
            from nvfuser_direct import (
                FusionDefinition,
                CommunicatorBackend,
                ParallelType,
                MemoryType,
            )
            from nvfuser_direct.pytorch_utils import torch_dtype_to_nvfuser_dtype
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "LinearColumnwise: fuser backend requested but nvfuser_direct is not available"
            ) from exc
        self._nvfuser = nvfuser
        self._FusionDefinition = FusionDefinition
        self._CommunicatorBackend = CommunicatorBackend
        self._ParallelType = ParallelType
        self._MemoryType = MemoryType
        self._torch_dtype_to_nvfuser_dtype = torch_dtype_to_nvfuser_dtype
        self._executor_cache: Dict[tuple, Any] = {}

    def _build_executor(
        self, dtype: torch.dtype, m: int, k: int, n: int, options: Dict[str, Any],
    ):
        nvfuser = self._nvfuser
        FusionDefinition = self._FusionDefinition
        CommunicatorBackend = self._CommunicatorBackend
        ParallelType = self._ParallelType
        MemoryType = self._MemoryType
        torch_dtype_to_nvfuser_dtype = self._torch_dtype_to_nvfuser_dtype

        d = self._world_size
        transport = options.get("transport", "cuda")
        comm_backend = CommunicatorBackend.cuda if transport == "cuda" else CommunicatorBackend.nccl
        algorithm = options.get("algorithm", "default")
        offset_by_rank = options.get("offset_stream_indexing_by_rank", False)

        use_symmetric = (
            comm_backend == CommunicatorBackend.cuda
            and not (algorithm == "p2p_pipeline" and offset_by_rank)
        )

        class _AgMatmulFusion(FusionDefinition):
            def definition(self_fd) -> None:
                nv_dtype = torch_dtype_to_nvfuser_dtype(dtype)
                self_fd.A = self_fd.define_tensor(
                    shape=[d, m // d, k], contiguity=True, dtype=nv_dtype,
                )
                self_fd.B = self_fd.define_tensor(
                    shape=[k, n], contiguity=True, dtype=nv_dtype,
                )
                self_fd.C = self_fd.ops.matmul(self_fd.A, self_fd.B)
                if use_symmetric:
                    self_fd.C.set_memory_type(MemoryType.symmetric)
                self_fd.add_output(self_fd.C)

            def multidevice_schedule(self_fd) -> None:
                mesh = nvfuser.multidevice.DeviceMesh(range(d))
                for tv in [self_fd.A, self_fd.B, self_fd.C]:
                    tv.set_device_mesh(mesh)
                self_fd.A.axis(0).parallelize(ParallelType.mesh_x)
                if algorithm == "p2p_pipeline":
                    self_fd.C.axis(0).parallelize(ParallelType.stream)

        multicast = options.get("multicast_protocol", "default")
        env_flags = (
            f"multicast_protocol({multicast})" if multicast != "default" else None
        )
        old_nv_enable = os.environ.get("NVFUSER_ENABLE")
        if env_flags is not None:
            os.environ["NVFUSER_ENABLE"] = env_flags
        try:
            fusion = _AgMatmulFusion()
            with fusion:
                fusion.definition()
                fusion.multidevice_schedule()

            params = nvfuser.multidevice.MultiDeviceExecutorParams()
            params.backend_type = comm_backend
            params.use_allocation_cache = True
            params.offset_stream_indexing_by_rank = offset_by_rank
            params.inter_stream_synchronization = options.get(
                "inter_stream_synchronization", False,
            )
            executor = nvfuser.multidevice.MultiDeviceExecutor(fusion.fusion, params)
        finally:
            if old_nv_enable is not None:
                os.environ["NVFUSER_ENABLE"] = old_nv_enable
            else:
                os.environ.pop("NVFUSER_ENABLE", None)

        return executor

    def _get_executor(
        self, dtype: torch.dtype, m: int, k: int, n: int, options: Dict[str, Any],
    ):
        key = (
            dtype, m, k, n,
            options.get("algorithm"),
            options.get("transport"),
            options.get("multicast_protocol"),
            options.get("offset_stream_indexing_by_rank"),
            options.get("inter_stream_synchronization"),
        )
        if key not in self._executor_cache:
            self._executor_cache[key] = self._build_executor(dtype, m, k, n, options)
        return self._executor_cache[key]

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        world_size: int,
        options: Dict[str, Any],
    ) -> torch.Tensor:
        input_2d = input.reshape(-1, input.shape[-1])
        m = input_2d.shape[0] * world_size
        executor = self._get_executor(
            input.dtype, m, input_2d.shape[1], weight.shape[1], options,
        )
        A = input_2d.unsqueeze(0)
        C = executor.run([A, weight])[0]
        result = C.reshape(m, weight.shape[1])
        if bias is not None:
            result = result + bias
        return result


class LinearColumnwise(nn.Module):
    """
    Column-wise linear with fused all-gather of the sharded input.

    Matches ``torch.ops.symm_mem.fused_all_gather_matmul`` semantics:
      - input  ``[m_local, k]``  is row-sharded (each rank holds m/world_size rows)
      - weight ``[k, n]``        is replicated
      - bias   ``[n]``           is replicated (optional)
      - output ``[m, n]``        is the full result after all-gather + matmul
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        backend: Optional[str] = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        heuristic: Optional[Heuristic] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if device is not None and device.type != "cuda":
            raise ValueError("LinearColumnwise: only CUDA devices are supported")
        self.in_features = in_features
        self.out_features = out_features
        backend = backend or "auto"
        self.backend = backend

        if backend == "auto":
            self._heuristic = heuristic or DEFAULT_HEURISTIC
            self.options = {}
        else:
            self._heuristic = None
            self.options = self._resolve_base_options(backend, options or {})
            self.backend = self.options["backend"]

        self._dist = _get_dist_context(self.backend, process_group)
        self._world_size = self._dist.world_size

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        pg, ws = self._dist.process_group, self._world_size
        if self.backend == "auto":
            self._backends = {"pytorch": _PytorchBackend(pg)}
            if self._is_fuser_available():
                self._backends["fuser"] = _FuserBackend(pg, ws)
        elif self.backend == "pytorch":
            self._backends = {"pytorch": _PytorchBackend(pg)}
        elif self.backend == "fuser":
            self._backends = {"fuser": _FuserBackend(pg, ws)}
        else:
            raise ValueError(f"LinearColumnwise: unsupported backend '{self.backend}'")

        self._cached_mkn = (-1, -1, -1)
        self._cached_call = None

    @staticmethod
    def _is_fuser_available() -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            import nvfuser_direct  # noqa: F401
        except Exception:
            return False
        return True

    @classmethod
    def _select_backend(cls, backend: str) -> str:
        if backend == "fuser" and not cls._is_fuser_available():
            return "pytorch"
        return backend

    @classmethod
    def _resolve_base_options(cls, backend: str, options: Dict[str, Any]) -> Dict[str, Any]:
        resolved = dict(options)
        requested_backend = resolved.get("backend", backend)
        resolved["backend"] = cls._select_backend(requested_backend)
        resolved.setdefault("transport", "cuda")
        resolved.setdefault("algorithm", None)
        resolved.setdefault("multicast_protocol", "default")
        resolved.setdefault("offset_stream_indexing_by_rank", False)
        resolved.setdefault("inter_stream_synchronization", False)
        return resolved

    @staticmethod
    def _resolve_runtime_options(
        m: int, in_features: int, out_features: int, options: Dict[str, Any]
    ) -> Dict[str, Any]:
        resolved = dict(options)
        if resolved.get("backend") != "fuser":
            return resolved
        if resolved.get("algorithm") is not None:
            return resolved
        mkn = m * in_features * out_features
        # Heuristic thresholds (tuned later): small < 1e8, medium < 1e10, else large
        if mkn < 1e8:
            resolved["algorithm"] = "default"
        elif mkn < 1e10:
            resolved["algorithm"] = "p2p_pipeline"
            resolved["offset_stream_indexing_by_rank"] = False
            resolved["inter_stream_synchronization"] = False
        else:
            resolved["algorithm"] = "p2p_pipeline"
            resolved["offset_stream_indexing_by_rank"] = True
            resolved["inter_stream_synchronization"] = True
        return resolved

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def _resolve_dispatch(self, m_local: int) -> None:
        k, n = self.in_features, self.out_features
        if self._heuristic is not None:
            config = self._heuristic.select(m_local, k, n)
            backend_key = config.backend
            if backend_key not in self._backends:
                backend_key = "pytorch"
            options = config.to_options()
            if _DDLP_DEBUG:
                print(
                    f"[DDLP] m_local={m_local} k={k} n={n} "
                    f"-> backend={backend_key} algorithm={config.algorithm} "
                    f"transport={config.transport} mcast={config.multicast_protocol} "
                    f"offset_by_rank={config.offset_stream_indexing_by_rank} "
                    f"inter_stream_sync={config.inter_stream_synchronization}"
                )
        else:
            backend_key = self.backend
            options = self._resolve_runtime_options(
                m_local, self.in_features, self.out_features, self.options
            )

        self._cached_mkn = (m_local, k, n)
        backend = self._backends[backend_key]
        weight, bias, ws = self.weight, self.bias, self._world_size
        if backend_key == "fuser":
            opts = options
            self._cached_call = lambda inp: backend.forward(inp, weight, bias, ws, opts)
        else:
            self._cached_call = lambda inp: backend.forward(inp, weight, bias, ws)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        m_local = int(input.numel() // input.shape[-1])
        if (m_local, self.in_features, self.out_features) != self._cached_mkn:
            self._resolve_dispatch(m_local)
        return self._cached_call(input)

