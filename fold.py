from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .core import DelegatingInnerOptimizer


class FoldingInnerOptimizer(DelegatingInnerOptimizer):
    """
    Inner optimizer that appends a folding function value to state.

    For example, if intermediate values are large but are used in a loss
    function which greatly reduces the size, then calling `minimize` with
    `return_intermediate=True` may be infeasible. This optimizer allows a total
    loss to be calculated based on all intermediate values, without actually
    returning the intermediate inference values.

    See `tf_unopt/example/loss_folding.py` for example.
    """
    def __init__(
            self, base_optimizer, init_fold_state_fn, reduce_fn,
            name='inner_folding'):
        """
        Construct the optimizer based on a base version and folding functions.

        Args:
            `base_optimizer`: base implementation to use.
            `init_fold_state_fn`: initial fold state function
                maps x0 -> fold_state.
            `reduce_fn`: function mapping (fold_state, x) -> fold_state.
            `name`: used in namespacing.
        """
        self._init_fold_state_fn = init_fold_state_fn
        self._reduce_fn = reduce_fn
        super(FoldingInnerOptimizer, self).__init__(base_optimizer, name)

    def zero_state(self, x):
        """
        Get the initial state.

        This is the base optimizer's state with the initial fold state
        appended.
        """
        base_state = super(FoldingInnerOptimizer, self).zero_state(x)
        fold_state = self._init_fold_state_fn(x)
        if not isinstance(fold_state, tf.Tensor):
            raise NotImplementedError('Only tensor `fold_state`s supported')
        return base_state + (fold_state,)

    def apply_gradients(self, grads, x, state):
        """
        Apply gradients to solution tensors.

        This is equivalent to the base implementation's `apply_gradients` using
        the state computed by the base implementation. In addition, the fold
        state is updated.
        """
        fold_state = state[-1]
        state = state[:-1]
        x, state = super(FoldingInnerOptimizer, self).apply_gradients(
            grads, x, state)
        fold_state = self._reduce_fn(fold_state, x)
        state = state + (fold_state,)
        return x, state
