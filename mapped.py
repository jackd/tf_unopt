from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .core import DelegatingInnerOptimizer


class MappedInnerOptimizer(DelegatingInnerOptimizer):
    def __init__(self, base_optimizer, pre_map_fn=None, post_map_fn=None,
                 name=None):
        """
        Create an optimizer based on another that maps values

        i.e. apply_gradients looks like
        ```
        grads, x, state = pre_map_fn(grads, x, state)
        x, state = base_optimizer.apply_gradients(grads, x, state)
        x, state = post_map_fn(x, state)
        return x, state
        ```

        None values for map functions mean the corresponding step is skipped.

        Args:
            `base_optimizer`: base implementation. Must be an `InnerOptimizer`.
            `pre_map_fn`: function mapping (grads, x, state) to modified
                (grads, x, state). Applied after computing gradients and before
                `base_optimizer`'s `apply_gradient` method.
            `post_map_fn`: function mapping (x, state) to modified (x, state).
                Applied after `base_optimizer.apply_gradients`
            `name`: optimizer name. If `None`, uses
                `"mapped_%s" % base_optimizer.name`.
        """
        self._pre_map_fn = pre_map_fn
        self._post_map_fn = post_map_fn
        if name is None:
            name = 'mapped_%s' % base_optimizer.name
        super(MappedInnerOptimizer, self).__init__(
            base_optimizer, name)

    def apply_gradients(self, grads, x, state):
        if self._pre_map_fn is not None:
            grads, x, state = self._pre_map_fn(grads, x, state)
        x, state = super(MappedInnerOptimizer, self).apply_gradients(
            grads, x, state)
        if self._post_map_fn is not None:
            x, state = self._post_map_fn(x, state)
        return x, state
