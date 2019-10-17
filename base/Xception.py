import os
import sys
import tensorflow as tf
import numpy as np
sys.path.append(os.path.abspath(".."))
from models.Xception import Xception
from visualize.model_test import Test_Model
slim = tf.contrib.slim

class Xception_use():
    def __init__(self,
                 xception_size,
                 data_format='channels_last',
                 dtype='float32',
                 is_classification=False,
                 num_classes=None,
                 reuse=False):
        """

        :param inPuts:
        :param resnet_size:
        :param data_format: For now, only support channels_last
        :param dtype:
        :param is_train:
        :param is_classification:
        """
        assert data_format == 'channels_last', 'Currently, only support data_format as \'channels_last\'.'
        self.xception_size = xception_size
        self.data_format = data_format
        self.dtype = dtype
        self.is_classification = is_classification
        self.block_stride=[1, 2, 2, 2]
        self.num_classes = num_classes
        self.reuse = reuse

    def forward(self):
        """

        :return:
        """
        model = Xception(
            self.xception_size,
            is_classification=self.is_classification,
            num_classes=self.num_classes,
            reuse=self.reuse)
        return model

def Xception_forward(inPuts, xception_size, is_classification=False, num_classes=None, is_train=False, data_format='channels_last'):
    """

    :param inPuts:
    :param xception_size:
    :param is_classification:
    :param num_classes:
    :param is_train:
    :param data_format:
    :return:
    """
    return Xception_use(xception_size=xception_size, is_classification=is_classification,
                        num_classes=num_classes, data_format=data_format).forward()(inPuts, is_train=is_train)

if __name__ == '__main__':
    model = Xception_use(xception_size=41, is_classification=False, num_classes=None).forward()
    Test_Model(model, input_size=(1, 800, 800, 3), speed_test=Test_Model, print_tensor=True)

#
# inPuts = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
# model = Xception_use(xception_size=41, is_classification=False, num_classes=None).forward()
# net, end_points = model(inPuts, is_train=False)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# data = np.ones(shape=(1, 800, 800, 3))
# feed_dict = {inPuts: data}
# # Test_Speed(sess, feed_dict, output)
# Print_Net(sess=sess)
# weight_path = 'D:\\Github\\TensorBox\\weights\\Xception\\xception_41_2018_05_09\\xception_41\\model.ckpt'
# saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='xception_41'))
# saver.restore(sess, weight_path)