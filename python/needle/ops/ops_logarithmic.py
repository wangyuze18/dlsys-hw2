from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z,axis=1,keepdims=True)
        return Z - (max_Z + array_api.log(array_api.sum(array_api.exp(Z-max_Z),axis=1,keepdims=True)))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        max_in = array_api.max(node.inputs[0].realize_cached_data(),axis=(1,),keepdims=True)
        exp_in = exp(node.inputs[0] - max_in)
        exp_sum = exp_in.sum(axes=(1,))
        return out_grad - (out_grad.sum(axes=(1,)) / exp_sum).reshape((input_shape[0],1)).broadcast_to(input_shape) * exp_in
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
 
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z,axis=self.axes,keepdims=True)
        return array_api.log(array_api.sum(array_api.exp(Z-max_Z),axis=self.axes)) + array_api.squeeze(max_Z,axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        max_in = array_api.max(node.inputs[0].realize_cached_data(),axis=self.axes,keepdims=True)
        exp_in = exp(node.inputs[0] - max_in)
        exp_sum = exp_in.sum(axes=self.axes)
        if self.axes == None:
          return (out_grad/exp_sum).broadcast_to(input_shape) * exp_in
        else:
          new_shape = [1 if i in self.axes else dim for i, dim in enumerate(input_shape)]
          return (out_grad/exp_sum).reshape(new_shape).broadcast_to(input_shape) * exp_in
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

