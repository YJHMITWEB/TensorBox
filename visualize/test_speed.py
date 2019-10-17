import tensorflow as tf
import numpy as np
import time

def Test_Speed(sess, feed_dict, output):
    """

    :param sess:
    :param feed_dict:
    :return:
    """
    start = time.time()
    test_round = 7
    each_round = 30
    for i in range(test_round * each_round + 1):
        sess.run(output, feed_dict=feed_dict)
        if i % each_round == 0 and i != 0:
            print('Test round [{}/{}], Model runs in {:.2f} fps...'.format(
                (i + 1) // each_round, test_round, each_round / (time.time() - start)))
            start = time.time()