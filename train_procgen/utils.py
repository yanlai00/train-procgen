import tensorflow as tf
import numpy as np

def observation_input(ob_space, batch_size=None, name='Ob'):
    from gym.spaces import Discrete, Box, MultiDiscrete
    from baselines.common.input import encode_observation
    assert isinstance(ob_space, Discrete) or isinstance(ob_space, Box) or isinstance(ob_space, MultiDiscrete), \
        'Baselines only deal with Discrete and Box observation spaces'
    dtype = ob_space.dtype
    if dtype == np.int8:
        dtype = np.uint8
    shape = (ob_space.shape[0], ob_space.shape[1], ob_space.shape[2])
    placeholder = tf.placeholder(shape=(batch_size,) + shape, dtype=dtype, name=name)
    return placeholder, encode_observation(ob_space, placeholder)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



