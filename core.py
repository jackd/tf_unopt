"""
Provides the abstract base class `InnerOptimizer`.

Implementations:
    - `tf_unopt.gradient_descent.InnerGradientDescentOptimizer`
    - `tf_unopt.momentum.InnerMomentumOptimizer`
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def is_tensor_tuple(x):
    """Checks whether x is a tuple and each entry is a tensor."""
    return isinstance(x, tuple) and all(isinstance(xi, tf.Tensor) for xi in x)


def as_tuple(val, nx, expected_type):
    """
    Get a tuple version of `val`.

    If `val` is a tuple, checks that it has length `nx` and each entry v
    `isinstance(v, expected_type)`.

    If `val` is not a tuple, returns a tuple repeating `val` `nx` times.
    """
    if isinstance(val, tuple):
        if len(val) != nx:
            raise ValueError(
                'Expected tuple of length %d, got %d' % (nx, len(val)))
        if not all(isinstance(v, type) for v in val):
            raise TypeError('Expected all to be %s' % str(expected_type))
        return val
    else:
        if not isinstance(val, expected_type):
            raise TypeError('Expected value to be %s' % str(expected_type))
        return tuple(val for _ in range(nx))


def assert_is_tensor_tuple(x, name='x'):
    """
    Check if `x` is a tuple of tensors.

    Raises a `ValueError` if not.
    """
    if not is_tensor_tuple(x):
        raise ValueError(
            '%s must be a tuple of tensors, got %s' % (name, str(x)))


def tensor_array_like(x, size):
    return tf.TensorArray(size=size, dtype=x.dtype, element_shape=size.shape)


def explicit_for_loop(body, x, n_steps):
    """
    Iterate over the `body` function explicitly for `n_steps`.

    Args:
        `body`: function mapping `n` tensors to a `n` tensors of the same form.
        `x`: initial inputs
        `n_steps`: number of iterations

    Returns:
        last output of `body`.
    """
    with tf.name_scope('explicit_while'):
        for i in range(n_steps):
            with tf.name_scope('step%d' % i):
                x = body(*x)
        return x


class InnerOptimizer(object):
    """
    Abstract base class for managing optimization as a network layer.

    The main function is `minimize`, which minimizes the function provided
    in the constructor.

    Concrete derived classes must implement
        * `apply_gradients`; and (optionally)
        * `zero_state`.

    If this constructor is not called then `get_loss` and property `name` must
        also be implemented.

    Implementations:
        - `tf_unopt.gradient_descent.InnerGradientDescentOptimizer`
        - `tf_unopt.momentum.InnerMomentumOptimizer`

    Example usage:
    ```python
    def f(x):
        return tf.reduce_sum(tf.square(x))

    optimizer = InnerGradientDescentOptimizer(f, learning_rate=0.1)
    x0 = tf.constant([0.5, 0.8])
    sols = optimizer.minimize(
        x, maximum_iterations=10, back_prop=True, return_intermediate=True)
    ```
    """

    def __init__(self, f, name='inner_optimizer'):
        """
        Create an instance for minimizing function `f`.

        The function `f` should take any number of tensor arguments. It should
        *not* create any variables. Functions which do create variables should
        either be refactored to remove variable creation from the function
        call itself, or be wrapped in `tf.variable_scope` with
        `reuse=tf.AUTO_REUSE` prior to use in this constructor. No error will
        be raised if not, but no guarantees are made about the behaviour of
        calls in this function is this is the case.

        Args:
            `f`: tensorflow function mapping any number of tensor arguments
                to a single tensor. This function will be minimized during
                calls to `step` and `minimize`. It should note create any
                variables of its own. See note above.
            `name`: for name-spacing
        """
        self._name = name
        self._f = f
        self._converged_fn = None

    @property
    def name(self):
        return self._name

    def compute_gradients(self, x):
        """
        Get the gradients of `f` (from constructor) for inputs `x`.

        Args:
            `x`: tuple of tensors, inputs to function `f` provided in
                constructor.
        Returns:
            `grads`: tuple of tensors representing gradients of f w.r.t inputs.
                Should be in the same order as inputs.
        """
        assert_is_tensor_tuple(x)
        with tf.name_scope('inner_gradients'):
            grads = tuple(tf.gradients(self.get_loss(x), x))
        return grads

    def apply_gradients(self, grads, x, state):
        """Apply gradients `grads` to inputs `x` for the given `state`."""
        raise NotImplementedError('Abstract method')

    def zero_state(self, x):
        """
        Get the initial state of the optimizer.

        Args:
            `x`: tuple of tensors. The number should correspond to the number
                of arguments of `f` supplied in the constructor.

        Returns:
            `state`: optimizer state, tuple of tensors. Can be an empty tuple
                (`()`) for optimizers that do not change state between calls to
                `step`.
        """
        return ()

    def step(self, x, state):
        """
        Perform a single step of optimization.

        Args:
            `x`: function inputs, tuple of tensors. The length of this should
                match the number of arguments to `f` supplied in the
                constructor.
            `state`: optimizer state, tuple of tensors
        Returns:
            `x, state`: function inputs/state with same structure as inputs
        """
        grads = self.compute_gradients(x)
        x, state = self.apply_gradients(grads, x, state)
        return x, state

    def get_loss(self, x):
        """Get the loss value associated with the function from constructor."""
        return self._f(*x)

    def minimize(
            self, x0, maximum_iterations, converged_fn=None, back_prop=True,
            return_intermediate=True, explicit_loop=False, return_state=False,
            **loop_kwargs):
        """
        Search for a minimum to f from x0 for `maximum_iterations` steps.

        Args:
            `x0`: initial solution approximation, tuple of tensors.
            `maximum_iterations`: maximum number of iterations to take.
            `converged_fn`: if provided, is evaluated each step and allows for
                early breaking. Is ignored if `explicit_loop` is `True`.
            `back_prop`: whether or not the solution should be
                back-propagatable. Ignored if `explicit_loop` is `True`.
            `return_intermediate`: whether or not to return all intermediate
                solutions.
            `explicit_loop`: whether or not to explicitly loop through the
                steps or use `tf.while_loop`. Note `converged` is not used
                if explicit_loop is True.
            `return_state`: if True, returns both the solution(s) `x` and the
                state.
            `**loop_kwargs`: other kwargs passed to loop function.

        Returns:
            tuple of solutions, one for each value of x0.
                If `include_intermediate` is True, each solution is a
                `tf.TensorArray`, otherwise it is a `tf.Tensor`.
        """
        assert_is_tensor_tuple(x0, 'x0')
        x = x0
        with tf.name_scope('%s_minimize' % self.name):
            state = self.zero_state(x0)
            if explicit_loop:
                # less memory requirements when unrolled manually?
                def while_loop(cond, body, x):
                    return explicit_for_loop(
                        body, x, n_steps=maximum_iterations, **loop_kwargs)

            else:
                major, minor, patch = tf.__version__.split('.')
                major, minor = int(major), int(minor)
                if major > 1 or major == 1 and minor > 4:
                    def while_loop(cond, body, x):
                        return tf.while_loop(
                            cond, body, x,
                            maximum_iterations=maximum_iterations,
                            back_prop=back_prop, **loop_kwargs)
                else:
                    def while_loop(cond, body, x):
                        # no maximum_
                        # more verbose than using maximum_iterations
                        # but compatible with older versions of tensorflow.
                        step = 0

                        def cond2(step, *args):
                            return tf.logical_and(
                                cond(*args), step < maximum_iterations)

                        def body2(step, *args):
                            return (step + 1,) + tuple(body(*args))

                        out = tf.while_loop(
                            cond2, body2, (step,) + x,
                            back_prop=back_prop, **loop_kwargs)
                        return out[1:]

            nx = len(x)
            ns = len(state)
            if return_intermediate:
                solutions = tuple(tf.TensorArray(
                    tf.float32, size=maximum_iterations) for _ in x)

                def unpack(args):
                    x = args[:nx]
                    state = args[nx:nx+ns]
                    solutions = args[nx+ns:-1]
                    i = args[-1]
                    return x, state, solutions, i

                def pack(x, state, sol, i):
                    return x + state + sol + (i,)

                def cond(*args):
                    if converged_fn is None:
                        return True
                    else:
                        x, state, solutions, i = unpack(args)
                        return tf.logical_not(converged_fn(x))

                def body(*args):
                    x, state, solutions, i = unpack(args)
                    x, state = self.step(x, state)
                    solutions = tuple(
                        sol.write(i, xi) for sol, xi in zip(solutions, x))
                    return pack(x, state, solutions, i + 1)

                out = while_loop(
                    cond, body,
                    pack(x, state, solutions, 0))
                x, state, solutions, i = unpack(out)
                if return_state:
                    return solutions, state
                else:
                    return solutions
            else:

                def unpack(args):
                    x = args[:nx]
                    state = args[nx:]
                    return x, state

                def pack(x, state):
                    return x + state

                def cond(*args):
                    if converged_fn is None:
                        return True
                    else:
                        x, state = unpack(args)
                        return tf.logical_not(converged_fn(x))

                def body(*args):
                    x, state = unpack(args)
                    x, state = self.step(x, state)
                    out = pack(x, state)
                    if len(out) == 1:
                        return out[0]
                    return out

                out = while_loop(cond, body, pack(x, state))
                if isinstance(out, tf.Tensor):
                    out = (out,)
                x, state = unpack(out)
                if return_state:
                    return x, state
                else:
                    return x


class DelegatingInnerOptimizer(InnerOptimizer):
    """
    Base class for InnerOptimizers based on other optimizers.

    Default implementation redirects core methods:
        * `get_loss`;
        * `zero_state`;
        * `compute_gradients`; and
        * `apply_gradients`

    While the class can be instantiated, its designed to be extended with
    some of the core methods overriden.

    See `MappedInnerOptimizer` for an example.
    """

    def __init__(self, base_optimizer, name):
        """Initialize with a base optimizer and name."""
        if not isinstance(base_optimizer, InnerOptimizer):
            raise TypeError(
                'base_optimizer must be an `InnerOptimizer`, '
                'got instance of type `%s`' % type(base_optimizer))
        self._base = base_optimizer
        self._name = name

    def get_loss(self, x):
        return self._base.get_loss(x)

    def zero_state(self, x):
        return self._base.zero_state(x)

    def compute_gradients(self, x):
        return self._base.compute_gradients(x)

    def apply_gradients(self, grads, x, state):
        return self._base.apply_gradients(grads, x, state)
