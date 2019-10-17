import tensorflow as tf

def Mean(inPuts, axis, keep_dim=True):
    """

    :param inPuts:
    :param axis:
    :param keep_dim:
    :return:
    """
    assert isinstance(axis, int) or isinstance(axis, tuple), 'Mean axis must be either an int or a tuple.'
    if isinstance(axis, tuple):
        axis = list(axis)

    return tf.reduce_mean(inPuts, axis=axis, keep_dims=keep_dim)