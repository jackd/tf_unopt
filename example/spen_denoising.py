#!/usr/bin/python
"""
Basic implementation of image denoising network.

Based on DeepPrior3 network of:

End-to-End Learning for Structured Prediction Energy Networks
Belanger, Yang and McCallum
https://arxiv.org/abs/1703.05667
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
sys.path.append(
    os.path.realpath(os.path.join(os.path.dirname(__file__), '../..')))


def get_inner_optimizer(fn, initial_learning_rate=0.1, initial_momentum=0.5):
    """Get an `InnerMomentumOptimizer` with learned optimization params."""
    from tf_unopt.momentum import InnerMomentumOptimizer
    learning_rate = tf.get_variable(
        'inner_learning_rate', dtype=tf.float32,
        initializer=initial_learning_rate)
    momentum = tf.get_variable(
        'inner_momentum', dtype=tf.float32, initializer=initial_momentum)
    return InnerMomentumOptimizer(
        fn, learning_rate=learning_rate, momentum=momentum)


def get_learned_loss_network(image):
    with tf.name_scope('dnn'):
        image = tf.layers.conv2d(image, 32, 7, activation=tf.nn.softplus)
        image = tf.layers.conv2d(image, 32, 7, activation=tf.nn.softplus)
        image = tf.layers.conv2d(image, 1, 1)
        out = tf.reduce_mean(image)
    return out


def change_loss(inferred, original):
    return tf.reduce_sum(tf.square(inferred - original))


def denoise_image(noisy_image, n_steps=3):
    """
    Get the solutions from minimizing combined change_loss and network loss.

    Args:
        noisy_image: (batch_size, h, w, c) image

    Returns:
        `tf.TensorArray`, size (n_steps,), each entry of same shape as
            `noisy_image`.
    """

    def f(image):
        with tf.variable_scope('spen_loss', reuse=tf.AUTO_REUSE):
            weight = tf.get_variable(
                    'learned_weighting', dtype=tf.float32, initializer=1.0)
            loss = change_loss(image, noisy_image) + \
                weight*get_learned_loss_network(image)
        return loss

    optimizer = get_inner_optimizer(f)
    sol, = optimizer.minimize(
        (noisy_image,),
        maximum_iterations=n_steps,
        back_prop=True,
        return_intermediate=True)

    return sol


noisy_image = tf.placeholder(shape=(None, 224, 224, 1), dtype=tf.float32)
inference_array = denoise_image(noisy_image)

vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for v in vars:
    print(v.name)
print('Total: %d variables' % len(vars))
