#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import os
sys.path.append(
    os.path.realpath(os.path.join(os.path.dirname(__file__), '../..')))

from tf_unopt.gradient_descent import InnerGradientDescentOptimizer  # NOQA
from tf_unopt.momentum import InnerMomentumOptimizer  # NOQA
from tf_unopt.mapped import MappedInnerOptimizer  # NOQA

use_momentum = True
# use_momentum = False
kwargs = dict(
    maximum_iterations=10,
    back_prop=True,
    # back_prop=False,
    return_intermediate=True,
    # return_intermediate=False,
    # explicit_loop=True,
    explicit_loop=False,
)

# gradient_clip_val = None
gradient_clip_val = 0.4


def f(x):
    r = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1))
    xi = tf.unstack(x, axis=-1)[-1]
    s = tf.cos(5*r) / 5 + r/2 + tf.abs(xi) / 3
    return s


if use_momentum:
    opt = InnerMomentumOptimizer(f, 0.2, 0.9)
else:
    opt = InnerGradientDescentOptimizer(f, 0.2)


if gradient_clip_val is not None:
    def clip(grads, x, state):
        grads = tuple(
            tf.clip_by_value(g, -gradient_clip_val, gradient_clip_val)
            for g in grads)
        return grads, x, state
    opt = MappedInnerOptimizer(opt, clip, name='clipped_%s' % opt.name)

x0 = tf.constant([1.5, 1.5], dtype=tf.float32),
n_steps = 10

sol, = opt.minimize(x0, **kwargs)
if kwargs['return_intermediate']:
    sol = sol.stack()

grid = tf.meshgrid(
    tf.linspace(-2.0, 2.0, 51), tf.linspace(-2.0, 2.0, 51), indexing='ij')
grid = tf.stack(grid, axis=-1)
f_vals = f(grid)
f0_val = f(x0)
sol_vals = f(sol)


with tf.Session() as sess:
    g, f, f0, s, sv = sess.run((grid, f_vals, f0_val, sol, sol_vals))


def vis(g, f, f0, s, sv):
    from mayavi import mlab
    x, y = s.T
    z = sv
    mlab.points3d(x, y, z, scale_factor=0.2, scale_mode='none')
    x, y = (g[..., i] for i in range(2))
    mlab.surf(x, y, f)
    print(f0, sv)
    mlab.show()


vis(g, f, f0, s, sv)
