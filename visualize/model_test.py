import tensorflow as tf
import numpy as np
import time
from visualize.print_tensor import Print_Net
from visualize.test_speed import Test_Speed

def Test_Model(model, input_size=(1, 800, 800, 3), speed_test=True, print_tensor=True):
    """

    :param model:
    :param input_size:
    :param speed_test:
    :param print_tensor:
    :return:
    """
    input_size = list(input_size)
    assert len(input_size) == 4, 'Test inputs should have exactly 4 dimensions.'
    for i, dim in enumerate(input_size):
        if dim <= 0:
            raise ValueError('Test inputs should not have 0 size in dim %d.' % i)
    inPuts = tf.placeholder(dtype=tf.float32, shape=[input_size[0], input_size[1], input_size[2], input_size[3]])
    output = model(inPuts, is_train=False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    data = np.random.randn(input_size[0], input_size[1], input_size[2], input_size[3])
    feed_dict = {inPuts: data}

    if print_tensor:
        print('\n\nPrinting model structure...')
        Print_Net(sess)

    if speed_test:
        print('\n\nTesting inference speed...')
        Test_Speed(sess, feed_dict=feed_dict, output=output)