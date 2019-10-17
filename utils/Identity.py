import tensorflow as tf

def Identity(inPuts, name):
    """

    :param inPuts:
    :param name:
    :return:
    """
    return tf.identity(inPuts, name)