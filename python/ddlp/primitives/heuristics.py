from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class HeuristicConfig:
    """Configuration selected by a heuristic for a given problem size."""

    backend: str
    algorithm: str = "default"
    transport: str = "cuda"
    multicast_protocol: str = "default"
    offset_stream_indexing_by_rank: bool = False
    inter_stream_synchronization: bool = False

    def __post_init__(self) -> None:
        opts: Dict[str, Any] = {
            "backend": self.backend,
            "algorithm": self.algorithm,
            "transport": self.transport,
            "multicast_protocol": self.multicast_protocol,
            "offset_stream_indexing_by_rank": self.offset_stream_indexing_by_rank,
            "inter_stream_synchronization": self.inter_stream_synchronization,
        }
        object.__setattr__(self, "_options", opts)

    def to_options(self) -> Dict[str, Any]:
        return self._options  # type: ignore[attr-defined]


class Heuristic(ABC):
    """Interface for selecting backend and options based on problem dimensions.

    Subclass and override ``select`` to implement custom selection logic.
    """

    @abstractmethod
    def select(self, m: int, k: int, n: int) -> HeuristicConfig:
        """Return configuration for the given problem dimensions.

        Args:
            m: Local number of rows per rank (m_local = total_rows / world_size).
            k: Inner dimension (in_features).
            n: Output features (out_features, replicated).
        """
        ...


class DecisionTreeHeuristic(Heuristic):
    """Decision-tree heuristic fitted with DDLB on a single 8xH100 DGX node.

    Classes:
        0 - Fuser eager, CUDA transport, memcpy multicast
        1 - Fuser eager, CUDA transport, multimem multicast
        2 - Fuser p2p pipeline, CUDA transport
        3 - PyTorch (NCCL)
    """

    FUSER_EAGER_MEMCPY = HeuristicConfig(
        backend="fuser",
        algorithm="default",
        transport="cuda",
        multicast_protocol="memcpy",
    )
    FUSER_EAGER_MULTIMEM = HeuristicConfig(
        backend="fuser",
        algorithm="default",
        transport="cuda",
        multicast_protocol="multimem",
    )
    FUSER_P2P_PIPELINE = HeuristicConfig(
        backend="fuser",
        algorithm="p2p_pipeline",
        transport="cuda",
    )
    PYTORCH = HeuristicConfig(backend="pytorch")

    def select(self, m: int, k: int, n: int) -> HeuristicConfig:
        # Thresholds fitted with DDLB on a single 8xH100 DGX node; subject to change.
        mk = m * k
        if mk <= 25_165_824:
            if mk <= 6_291_456:
                return self.FUSER_EAGER_MULTIMEM
            return self.FUSER_EAGER_MULTIMEM if n <= 1536 else self.PYTORCH
        mnk = m * n * k
        if mnk <= 824_633_720_832:
            return self.FUSER_EAGER_MEMCPY
        return self.FUSER_P2P_PIPELINE


DEFAULT_HEURISTIC = DecisionTreeHeuristic()
