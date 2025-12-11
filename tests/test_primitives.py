import torch
import pytest
from ddlp.primitives import (
    SPTPRowParallelLinear,
    SPTPColumnParallelLinear,
    TPRowParallelLinear,
    MixtureOfExperts
)

def test_sp_tp_row_parallel_linear():
    B, M, K, N = 2, 4, 16, 8
    model = SPTPRowParallelLinear(K, N)
    x = torch.randn(B, M, K)
    # Just checking it runs without error via C++ binding
    out = model(x)
    assert out.shape == (B, M, N)

def test_sp_tp_column_parallel_linear():
    B, M, K, N = 2, 4, 16, 8
    model = SPTPColumnParallelLinear(K, N)
    x = torch.randn(B, M, K)
    out = model(x)
    assert out.shape == (B, M, N)

def test_moe():
    B, M, H = 2, 4, 16
    E = 4
    model = MixtureOfExperts(E, H)
    x = torch.randn(B, M, H)
    gate = torch.randn(B, M, E)
    out = model(x, gate)
    assert out.shape == (B, M, H)



