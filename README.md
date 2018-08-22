## [tf_unopt](https://github.com/jackd/tf_unopt)
[Tensorflow](https://tensorflow.org) implementation of unrolled optimization.

## Setup
```
cd /path/to/parent_dir
export PYTHONPATH=$PYTHONPATH:/path/to.parent_dir
git clone https://github.com/jackd/tf_unopt.git
```

## Usage
Unlike standard optimization that occurs in tensorflow which involves modifying values inside `tf.Variable`s, `InnerOptimizer`s in this repository unroll optimization update steps, returning updated approximate solutions to minimization problems.

These solutions are differentiable with respect to their inputs (assuming the optimized function is), so can be used *within* a standard neural network architecture.

### Construction
Optimizer construction is done by providing a function that maps input tensors to a tensor output, along with optimization hyperparameters (which can be learned by the outer optimizer).

```
def f(x, y):
    return tf.reduce_sum(tf.square(x)) + tf.reduce_sum(x*y)

learning_rate = tf.get_variable(
    'learning_rate', initializer=0.1, dtype=tf.float32)
momentum = tf.get_variable(
    'momentum', initializer=0.5, dtype=tf.float32)
optimizer = tf_unopt.momentum.MomentumOptimizer(f, learning_rate, momentum)
```

If the function to be optimized is not a scalar, it's sum is optimized as per the behaviour of `tf.gradients`.

### Optimization
Optimization can be done either step by step:
```
args = x, y
state = optimizer.zero_state(args)
for i in range(n_steps):
    args, state = optimizer.step(args)
x_opt, y_opt = args
```
or via the `minimize` method:
```
args = x, y
x, y = optimizer.minimize((x, y), n_steps, return_intermediate=True)
# Returned values are TensorArrays
x = x.stack()
y = y.stack()
```
See [examples](./examples) and class documentation for more, or tensorflow documentation on [tf.TensorArray](https://www.tensorflow.org/api_docs/python/tf/TensorArray)s.

### Implementing your own InnerOptimizer
To implement your own `InnerOptimizer`, you must implement `step` and optionally `zero_state`. See [InnerGradientDescentOptimizer](./gradient_descent.py) and [MomentumOptimizer](./momentum_optimizer.py) for examples.

## Examples
In the [example]('./exmaple') directory you'll find:
* a [toy example]('./example/simple.py'); and
* more complete [network setup]('./example/spen_denoising.py') based on [Structured Prediction Energy Networks](https://arxiv.org/abs/1703.05667) with learnable inner-optimization variables.
