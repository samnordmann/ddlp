# pylint: disable=too-many-instance-attributes
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math
import torch
from torch import nn
from torch.nn import functional as F



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

    raise ValueError(f"LinearColumnwise: unsupported backend '{backend}'")


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


class _FuserBackend:
    def __init__(self, process_group: Optional[torch.distributed.ProcessGroup]):
        self.process_group = process_group
        try:
            import nvfuser_direct as nvfuser
            from nvfuser_direct import FusionDefinition
            from nvfuser_direct.pytorch_utils import torch_dtype_to_nvfuser_dtype
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "LinearColumnwise: fuser backend requested but nvfuser_direct is not available"
            ) from exc
        self._nvfuser = nvfuser
        self._FusionDefinition = FusionDefinition
        self._torch_dtype_to_nvfuser_dtype = torch_dtype_to_nvfuser_dtype
        self._fusion_cache = {}

    def _get_executor(
        self,
        dtype: torch.dtype,
        m: int,
        k: int,
        n: int,
        has_bias: bool,
    ):
        key = (dtype, m, k, n, has_bias)
        if key in self._fusion_cache:
            return self._fusion_cache[key]

        FusionDefinition = self._FusionDefinition
        torch_dtype_to_nvfuser_dtype = self._torch_dtype_to_nvfuser_dtype

        class _LocalLinearFusion(FusionDefinition):
            def __init__(
                self,
                dtype: torch.dtype,
                m: int,
                k: int,
                n: int,
                has_bias: bool,
            ):
                super().__init__()
                self.dtype = dtype
                self.m = m
                self.k = k
                self.n = n
                self._has_bias = has_bias

            def definition(self) -> None:
                self.A = self.define_tensor(
                    shape=[self.m, self.k],
                    contiguity=True,
                    dtype=torch_dtype_to_nvfuser_dtype(self.dtype),
                )
                self.B = self.define_tensor(
                    shape=[self.k, self.n],
                    contiguity=True,
                    dtype=torch_dtype_to_nvfuser_dtype(self.dtype),
                )
                if self._has_bias:
                    self.Bias = self.define_tensor(
                        shape=[self.n],
                        contiguity=True,
                        dtype=torch_dtype_to_nvfuser_dtype(self.dtype),
                    )
                    self.C_matmul = self.ops.matmul(self.A, self.B)
                    self.C = self.ops.add(self.C_matmul, self.Bias)
                else:
                    self.C_matmul = self.ops.matmul(self.A, self.B)
                    self.C = self.C_matmul
                self.add_output(self.C)

        fusion = _LocalLinearFusion(
            dtype,
            m,
            k,
            n,
            has_bias,
        )
        with fusion:
            fusion.definition()
        self._fusion_cache[key] = fusion
        return fusion

    def _get_rank(self) -> int:
        if self.process_group is None:
            return torch.distributed.get_rank()
        return torch.distributed.get_rank(self.process_group)

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        world_size: int,
        options: Dict[str, Any],
    ) -> torch.Tensor:
        input_2d = input.reshape(-1, input.shape[-1])
        m = input_2d.shape[0]
        _ = self._get_rank()
        weight_t = weight.t().contiguous()
        inputs = [input_2d.contiguous(), weight_t]
        has_bias = bias is not None
        if has_bias:
            inputs.append(bias.contiguous())
        executor = self._get_executor(
            input.dtype,
            m,
            input_2d.shape[1],
            weight_t.shape[1],
            has_bias,
        )
        local_out = executor.execute(inputs)[0].reshape(*input.shape[:-1], weight.shape[0])
        gathered_out = [torch.empty_like(local_out) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_out, local_out, group=self.process_group)
        return torch.cat(gathered_out, dim=-1)


class _TransformerEngineBackend:
    def __init__(
        self,
        in_features: int,
        out_features: int,
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
        self._te = te
        self.process_group = process_group
        self._world_size = torch.distributed.get_world_size(process_group)
        self._rank = torch.distributed.get_rank(process_group)
        self._device = weight.device
        self._bias_enabled = bias
        self._ub_shape = None
        self._tp_group = process_group if process_group is not None else torch.distributed.group.WORLD
        self._ub_name = "qkv"
        self.linear = te.Linear(
            in_features,
            out_features,
            bias=bias,
            device=self._device,
            params_dtype=weight.dtype,
            sequence_parallel=True,
            parallel_mode="column",
            tp_group=self._tp_group,
            ub_overlap_ag=True,
            tp_size=self._world_size,
            ub_name=self._ub_name,
        )
        if hasattr(self.linear, "set_tensor_parallel_group"):
            self.linear.set_tensor_parallel_group(self._tp_group)
        if self.linear.weight.shape != weight.shape:
            raise RuntimeError(
                "LinearColumnwise: transformer_engine weight shape mismatch with module weight"
            )
        self.linear.weight = weight
        if bias:
            if self.linear.bias.shape != bias_param.shape:
                raise RuntimeError(
                    "LinearColumnwise: transformer_engine bias shape mismatch with module bias"
                )
            self.linear.bias = bias_param

    def _ensure_ub(self, m: int, k: int, dtype: torch.dtype) -> None:
        if self._ub_shape == (m, k):
            return
        if self._ub_shape is not None:
            try:
                self._te.module.base.destroy_ub()
            except Exception:
                pass
        self._te.module.base.initialize_ub(
            shape=[m, k],
            tp_size=self._world_size,
            use_fp8=False,
            dtype=dtype,
            ub_cfgs=None,
            bootstrap_backend="nccl",
        )
        self._ub_shape = (m, k)

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        world_size: int,
        options: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        input_2d = input.reshape(-1, input.shape[-1])
        m = input_2d.shape[0]
        if m % world_size != 0:
            raise ValueError("LinearColumnwise: m must be divisible by world_size for TE backend")
        self._ensure_ub(m, input_2d.shape[1], input.dtype)
        input_sharded = (
            input_2d.view(world_size, m // world_size, input_2d.shape[1])[self._rank : self._rank + 1]
            .contiguous()
        )
        input_sharded = input_sharded.reshape(m // world_size, input_2d.shape[1])
        local_out = self.linear(input_sharded)
        if local_out.shape[0] == m:
            gathered_out = [torch.empty_like(local_out) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_out, local_out, group=self.process_group)
            full_out = torch.cat(gathered_out, dim=-1)
        elif local_out.shape[0] == m // world_size:
            gathered_seq = [torch.empty_like(local_out) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_seq, local_out, group=self.process_group)
            seq_full = torch.cat(gathered_seq, dim=0)
            gathered_feat = [torch.empty_like(seq_full) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_feat, seq_full, group=self.process_group)
            full_out = torch.cat(gathered_feat, dim=-1)
        else:
            raise RuntimeError("LinearColumnwise: unexpected TE output shape for reconstruction")
        return full_out.reshape(*input.shape[:-1], full_out.shape[-1])


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
        options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if device is not None and device.type != "cuda":
            raise ValueError("LinearColumnwise: only CUDA devices are supported")
        self.in_features = in_features
        self.out_features = out_features
        self.options = self._resolve_base_options(backend, options or {})
        self.backend = self.options["backend"]

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
                out_features,
                bias,
                self.weight,
                self.bias,
                self._dist.process_group,
            )
        else:
            raise ValueError(f"LinearColumnwise: unsupported backend '{self.backend}'")

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
        if backend in ("auto", "fuser"):
            return "fuser" if cls._is_fuser_available() else "pytorch"
        return backend

    @classmethod
    def _resolve_base_options(cls, backend: str, options: Dict[str, Any]) -> Dict[str, Any]:
        resolved = dict(options)
        requested_backend = resolved.get("backend", backend)
        resolved["backend"] = cls._select_backend(requested_backend)
        resolved.setdefault("transport", "cuda")
        resolved.setdefault("algorithm", None)
        resolved.setdefault("offset_stream_indexing_by_rank", False)
        resolved.setdefault("inter_stream_synchronization", False)
        return resolved

    @staticmethod
    def _resolve_runtime_options(
        input: torch.Tensor, in_features: int, out_features: int, options: Dict[str, Any]
    ) -> Dict[str, Any]:
        resolved = dict(options)
        if resolved.get("backend") != "fuser":
            return resolved
        if resolved.get("algorithm") is not None:
            return resolved
        m = int(input.numel() // in_features)
        k = in_features
        n = out_features
        mkn = m * k * n
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
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_cuda or not self.weight.is_cuda:
            raise RuntimeError("LinearColumnwise: only CUDA tensors are supported")
        if self.bias is not None and not self.bias.is_cuda:
            raise RuntimeError("LinearColumnwise: bias must be on CUDA")
        runtime_options = self._resolve_runtime_options(
            input, self.in_features, self.out_features, self.options
        )
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
        if self.backend == "fuser":
            return self._backend.forward(input, self.weight, self.bias, world_size, runtime_options)
        return self._backend.forward(input, self.weight, self.bias, world_size)

