"""Gradient descent with momentum implementation of InnerOptimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .gradient_descent import InnerGradientDescentOptimizer
from .core import assert_is_tensor_tuple, as_tuple


class InnerMomentumOptimizer(InnerGradientDescentOptimizer):
    """InnerOptimizer implementation of gradient descent with momentum."""
    def __init__(self, f, learning_rate, momentum, name='inner_momentum'):
        """
        Args:
            `f`: function to miminize.
            `learning_rate`: python number, tensor, or tuple of
                scalars/tensors. If a tuple, the number of elements should
                match the number of arguments required by `f`.
            `momentum`: python number, tensor, or tuple of scalars/tensors.
                If a tuple, the number of elements should match the number of
                arguments required by `f`.
        """
        self._momentum = momentum
        super(InnerMomentumOptimizer, self).__init__(f, learning_rate, name)

    @property
    def momentum(self):
        """Getter for value provided in constructor."""
        return self._momentum

    def zero_state(self, x):
        assert_is_tensor_tuple(x, 'x')
        state = tuple(tf.zeros_like(xi) for xi in x)
        return state

    def _update(self, x, grads, state):
        nx = len(x)
        types = (int, float, tf.Tensor, tf.Variable)
        lrs = lrs = as_tuple(self.learning_rate, nx, types)
        momentums = as_tuple(self.momentum, nx, types)
        ret_state = []
        ret_x = []
        for (xi, grad, s, lr, m) in zip(x, grads, state, lrs, momentums):
            acc = m * s + grad
            xi = xi - lr*acc
            ret_state.append(acc)
            ret_x.append(xi)
        return tuple(ret_x), tuple(ret_state)
