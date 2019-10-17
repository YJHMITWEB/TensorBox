import tensorflow as tf
from utils.Identity import Identity
from utils.Padding import Free_Padding

def MaxPool2D(inPuts, kernel_size, stride, padding='s', data_format='NHWC'):
    """

    :param inPuts:
    :param kernel_size:
    :param stride:
    :param padding:
    :param data_format:
    :return:
    """
    assert isinstance(padding, str) or isinstance(padding, int), \
        'Padding must be either \'s\' or \'v\' or an int.'

    if padding == 's':
        inPuts = tf.layers.max_pooling2d(inputs=inPuts, pool_size=kernel_size, strides=stride, padding='SAME',
                                         data_format=data_format)
    elif padding == 'v':
        inPuts = tf.layers.max_pooling2d(inputs=inPuts, pool_size=kernel_size, strides=stride, padding='VALID',
                                         data_format=data_format)
    elif isinstance(padding, int):
        inPuts = Free_Padding(inPuts, pad_size=padding, data_format=data_format)
        inPuts = tf.layers.max_pooling2d(inputs=inPuts, pool_size=kernel_size, strides=stride, padding='VALID',
                                         data_format=data_format)
    else:
        raise ValueError('Padding value is invalid.')

    return inPuts