import tensorflow as tf

def Remove_axis(inPuts, axis):
    """

    :param inPuts:
    :param axis:
    :return:
    """
    assert isinstance(axis, int) or isinstance(axis, tuple), 'Remove Axis must be either an int or a tuple'

    if isinstance(axis, tuple):
        axis = list(axis)

    return tf.squeeze(inPuts, axis=axis)