#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
# tf.enable_eager_execution()

maximum_iterations = 5
back_prop = True
loop_kwargs = {}


def while_loop_base(cond, body, x):
    return tf.while_loop(
        cond, body, x, maximum_iterations=maximum_iterations,
        back_prop=back_prop, **loop_kwargs)


def while_loop_alt(cond, body, x):
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


def cond(*args):
    x, y = args
    return y < 6


def body(*args):
    x, y = args
    x2 = y
    y2 = x + y
    return x2, y2


x0 = tf.constant(1, dtype=tf.int32)
x0 = (x0, x0)
base = while_loop_base(cond, body, x0)
alt = while_loop_alt(cond, body, x0)

with tf.Session() as sess:
    base, alt = sess.run((base, alt))

print(base)
print(alt)
