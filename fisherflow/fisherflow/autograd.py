"""
Automatic differentiation engine for Fisher Flow.
Extended from micrograd concepts to support Fisher Information computation.
"""

import numpy as np
from typing import List, Set, Optional, Callable


class Value:
    """
    A node in the computation graph with support for Fisher Information.
    Extended from micrograd-style Value to track second-order information.
    """
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self.fisher_diag = 0  # Diagonal Fisher Information
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
            # Fisher information propagates through addition
            self.fisher_diag += out.fisher_diag
            other.fisher_diag += out.fisher_diag
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            # Fisher diagonal approximation for multiplication
            self.fisher_diag += (other.data ** 2) * out.fisher_diag
            other.fisher_diag += (self.data ** 2) * out.fisher_diag
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')
        
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
            # Fisher diagonal for power operation
            self.fisher_diag += ((other * self.data**(other-1))**2) * out.fisher_diag
        out._backward = _backward
        
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
            # Fisher through ReLU
            self.fisher_diag += ((out.data > 0) ** 2) * out.fisher_diag
        out._backward = _backward
        
        return out
    
    def exp(self):
        x = self.data
        out = Value(np.exp(x), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
            # Fisher through exponential
            self.fisher_diag += (out.data ** 2) * out.fisher_diag
        out._backward = _backward
        
        return out
    
    def log(self):
        out = Value(np.log(self.data), (self,), 'log')
        
        def _backward():
            self.grad += (1 / self.data) * out.grad
            # Fisher through logarithm
            self.fisher_diag += (1 / self.data ** 2) * out.fisher_diag
        out._backward = _backward
        
        return out
    
    def backward(self):
        """Compute gradients via backpropagation."""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    
    def backward_fisher(self, fisher_value=1.0):
        """
        Compute diagonal Fisher Information via backpropagation.
        This approximates the Fisher Information Matrix diagonal elements.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.fisher_diag = fisher_value
        for v in reversed(topo):
            v._backward()
    
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


class Module:
    """Base class for neural network modules with Fisher Information support."""
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
            p.fisher_diag = 0
    
    def parameters(self):
        return []
    
    def accumulate_fisher(self, data_loader, loss_fn, damping=1e-4):
        """
        Accumulate empirical Fisher Information from data.
        
        Args:
            data_loader: Iterator over data batches
            loss_fn: Loss function that returns a Value
            damping: Small value for numerical stability
        """
        fisher_accumulator = {p: 0 for p in self.parameters()}
        n_samples = 0
        
        for batch in data_loader:
            self.zero_grad()
            
            # Forward pass and compute loss
            loss = loss_fn(batch, self)
            
            # Compute gradients
            loss.backward()
            
            # Accumulate squared gradients (diagonal Fisher approximation)
            for p in self.parameters():
                fisher_accumulator[p] += p.grad ** 2
            
            n_samples += 1
        
        # Average and store Fisher diagonal
        for p in self.parameters():
            p.fisher_diag = fisher_accumulator[p] / n_samples + damping


class Neuron(Module):
    """A single neuron with Fisher Information tracking."""
    
    def __init__(self, nin, nonlin=True):
        self.w = [Value(np.random.randn() * 0.1) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
    
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act
    
    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    """A layer of neurons with Fisher Information support."""
    
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """Multi-layer Perceptron with Fisher Flow support."""
    
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) 
                      for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]