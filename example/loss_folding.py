#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import os
sys.path.append(
    os.path.realpath(os.path.join(os.path.dirname(__file__), '../..')))

from tf_unopt.momentum import InnerMomentumOptimizer  # NOQA
from tf_unopt.fold import FoldingInnerOptimizer  # NOQA
kwargs = dict(
    maximum_iterations=10,
    back_prop=True,
    return_intermediate=False,
    explicit_loop=False,
    return_state=True
)


def f(x):
    r = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1))
    xi = tf.unstack(x, axis=-1)[-1]
    s = tf.cos(5*r) / 5 + r/2 + tf.abs(xi) / 3
    return s


decay = 0.9


def fold(old_fold_state, x):
    return decay * old_fold_state + f(x[0])


opt = InnerMomentumOptimizer(f, 0.2, 0.9)
opt = FoldingInnerOptimizer(opt, lambda x: f(x[0]), fold)
x0 = tf.constant([1.5, 1.5], dtype=tf.float32),
n_steps = 10

(sol,), state = opt.minimize(x0, **kwargs)
loss = state[-1]


with tf.Session() as sess:
    loss = sess.run(loss)


print(loss)
