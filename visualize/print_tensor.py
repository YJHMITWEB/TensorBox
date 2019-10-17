import tensorflow as tf

def Print_Net(sess):
    """

    :return:
    """
    variable_names = [v.name for v in tf.all_variables()]
    vs = sess.run(variable_names)
    for k, v in zip(variable_names, vs):
        print("Tensor: {}    Shape: {}".format(k, v.shape))