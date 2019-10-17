import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def BatchNorm(inPuts, is_train, data_format='NHWC'):
    """

    :param inputs:
    :param is_train:
    :param data_format:
    :return:
    """
    return tf.layers.batch_normalization(
              inputs=inPuts, axis=1 if data_format == 'channels_first' else 3,
              momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
              scale=True, training=is_train, fused=True)