from torch import nn
import torch
import ddlp._C.primitives as _cpp_primitives

class SPTPRowParallelLinear(nn.Module):
    """
    SP-TP-RowParallelLinear (Linear + Reduce-Scatter)
    Corresponds to the input of the second GEMM in an MLP block with Sequence Parallelism.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._impl = _cpp_primitives.SPTPRowParallelLinearImpl(in_features, out_features)
        
        # Register parameters (mocking for now, usually managed by implementation or passed in)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, input):
        # Delegate to C++ implementation
        return self._impl.forward(input, self.weight)

class SPTPColumnParallelLinear(nn.Module):
    """
    SP-TP-ColumnParallelLinear (AllGather + Linear)
    Corresponds to the input of the first GEMM in an MLP block with Sequence Parallelism.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._impl = _cpp_primitives.SPTPColumnParallelLinearImpl(in_features, out_features)
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
    def forward(self, input):
        return self._impl.forward(input, self.weight)

class TPRowParallelLinear(nn.Module):
    """
    TP-RowParallelLinear (Linear + AllReduce)
    Standard Tensor Parallelism without Sequence Parallelism.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._impl = _cpp_primitives.TPRowParallelLinearImpl(in_features, out_features)
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
    def forward(self, input):
        return self._impl.forward(input, self.weight)

class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts with Routing, Dispatch, and Combine.
    """
    def __init__(self, num_experts, hidden_size):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self._impl = _cpp_primitives.MixtureOfExpertsImpl(num_experts, hidden_size)
        
    def forward(self, input, gate):
        return self._impl.forward(input, gate)
