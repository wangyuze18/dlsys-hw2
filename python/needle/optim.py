"""Optimization module"""
from re import S
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
          if p.grad is not None:
            if p not in self.u.keys():
                self.u[p] = ndl.zeros_like(p.grad, requires_grad=False) 
            grad = p.grad.data + self.weight_decay * p.data
            self.u[p] = self.momentum * self.u[p] + (1 - self.momentum) * grad
            p.data = p.data - self.lr * self.u[p]
        
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        beta1_pow_t = self.beta1 ** self.t
        beta2_pow_t = self.beta2 ** self.t
        for p in self.params:
          if p.grad is not None:
            if p not in self.m.keys():
                self.m[p] = ndl.zeros_like(p.grad, requires_grad=False)
            if p not in self.v.keys():
                self.v[p] = ndl.zeros_like(p.grad, requires_grad=False) 
            grad = p.grad.data
            if self.weight_decay !=0:
              grad = grad + self.weight_decay * p.data
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * grad
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[p] / (1 - beta1_pow_t)
            v_hat = self.v[p] / (1 - beta2_pow_t)
            x_norm = m_hat / (v_hat ** 0.5 + self.eps)
            p.data = p.data - self.lr * x_norm
        ### END YOUR SOLUTION
