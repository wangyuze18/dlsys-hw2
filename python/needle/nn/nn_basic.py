"""The module.
"""
# from _typeshed import StrOrLiteralStr
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias == True:
          self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1,out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = X @ self.weight
        if self.bias != None:
          output = output + self.bias.broadcast_to(output.shape)
        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        size = math.prod([val for ind, val in enumerate(X.shape) if ind != 0])
        return X.reshape((X.shape[0],size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
          x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size , num_classes = logits.shape
        norm_logits = ops.logsoftmax(logits)
        return ( - init.one_hot(num_classes,y) * norm_logits).sum() / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim,device=device,dtype=dtype,requires_grad=True))
        self.bias = Parameter(init.zeros(dim,device=device,dtype=dtype,requires_grad=True))
        self.running_mean = init.zeros(dim,device=device,dtype=dtype,requires_grad=False)
        self.running_var = init.ones(dim,device=device,dtype=dtype,requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size , num_features = x.shape
        if self.training == True:
          mean = x.sum(axes=(0,)) / batch_size
          self.running_mean = (1 - self.momentum)*self.running_mean + self.momentum * mean.data
          mean = mean.broadcast_to(x.shape)
          var = ((x - mean) ** 2).sum(axes=(0,)) / batch_size
          self.running_var = (1 - self.momentum)*self.running_var + self.momentum * var.data
          var = var.broadcast_to(x.shape)
          x_norm = ((x-mean)/((var+self.eps) ** 0.5))
          return self.weight.broadcast_to(x.shape) * x_norm + self.bias.broadcast_to(x.shape)
        else:  
          x_norm = (x-self.running_mean)/((self.running_var + self.eps) ** 0.5)
          return self.weight.broadcast_to(x.shape) * x_norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim,device=device,dtype=dtype,requires_grad=True))
        self.bias = Parameter(init.zeros(dim,device=device,dtype=dtype,requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, num_features = x.shape
        mean = x.sum(axes=(1,)) / num_features
        mean = mean.reshape((batch_size,1)).broadcast_to(x.shape)
        var = ((x - mean) ** 2).sum(axes=(1,)) / num_features
        var = var.reshape((batch_size,1)).broadcast_to(x.shape)
        x_norm = (x - mean) / (var + self.eps) ** 0.5
        return self.weight.broadcast_to(x.shape) * x_norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.scale = 1.0 / (1.0 - p) if p < 1 else 1.0

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training or self.p == 0:
          return x

        mask = init.randb(*x.shape, p=1.0 - self.p, device=x.device, dtype=x.dtype) * self.scale
        return x * mask 
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
