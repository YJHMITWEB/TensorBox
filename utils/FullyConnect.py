import tensorflow as tf
from tensorflow.python.ops import init_ops

def FullyConnect(inPuts,
                 units,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 is_train=True,
                 name=None,
                 reuse=None
                 ):
    """

    :param inPuts:
    :param units:
    :param use_bias:
    :param kernel_initializer:
    :param bias_initializer:
    :param kernel_regularizer:
    :param bias_regularizer:
    :param activity_regularizer:
    :param kernel_constraint:
    :param bias_constraint:
    :param trainable:
    :param name:
    :param reuse:
    :return:
    """
    return tf.layers.dense(inPuts,
                           units=units,
                           use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           activity_regularizer=activity_regularizer,
                           kernel_constraint=kernel_constraint,
                           bias_constraint=bias_constraint,
                           trainable=is_train,
                           name=name,
                           reuse=reuse
                           )