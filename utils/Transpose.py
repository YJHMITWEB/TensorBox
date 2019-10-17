import tensorflow as tf

def Transpose(inPuts, axis):
    """

    :param inPuts:
    :param axis: A tuple denotes the target axis order
    :return:
    """
    axis = list(axis)
    return tf.transpose(inPuts, axis)
