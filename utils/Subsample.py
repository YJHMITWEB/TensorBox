import tensorflow as tf

def Subsample(inPuts, stride, data_format, name):
    """
    Unlike normal max_pooling, Subsample is a special ops which should not be changed.
    :param inPuts:
    :param stride:
    :param name:
    :return:
    """
    return tf.layers.max_pooling2d(inputs=inPuts, pool_size=1, strides=stride, padding='same', data_format=data_format,
                  name=name)