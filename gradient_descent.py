"""Gradient descent implementation of InnerOptimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .core import InnerOptimizer, as_tuple, assert_is_tensor_tuple


class InnerGradientDescentOptimizer(InnerOptimizer):
    def __init__(self, f, learning_rate, name='inner_gradient_descent'):
        """
        Args:
            `f`: function to miminize.
            `learning_rate`: python number, tensor, or tuple of
                scalars/tensors. If a tuple, the number of elements should
                match the number of arguments required by `f`.
        """
        self._learning_rate = learning_rate
        super(InnerGradientDescentOptimizer, self).__init__(f, name)

    @property
    def learning_rate(self):
        """Getter for value provided in constructor."""
        return self._learning_rate

    def step(self, x, state):
        assert_is_tensor_tuple(x)
        loss = self.get_loss(x)
        grads = tf.gradients(loss, x)
        x, state = self._update(x, grads, state)
        return x, state

    def _update(self, x, grads, state):
        types = (int, float, tf.Tensor, tf.Variable)
        lrs = as_tuple(self.learning_rate, len(x), types)
        x = tuple(xi - lr * grad for xi, lr, grad in zip(x, lrs, grads))
        return x, state
