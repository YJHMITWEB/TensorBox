import os
import sys
import tensorflow as tf
import numpy as np
sys.path.append(os.path.abspath(".."))
from models.ResNet import ResNet
from visualize.model_test import Test_Model

class ResNet_use():
    def __init__(self,
                 resnet_size,
                 data_format='channels_first',
                 dtype='float32',
                 is_classification=False,
                 num_classes=None):
        """

        :param inPuts:
        :param resnet_size:
        :param data_format:
        :param dtype:
        :param is_train:
        :param is_classification:
        """
        self.resnet_size = resnet_size
        self.data_format = data_format
        self.dtype = dtype
        self.is_classification = is_classification
        self.block_stride=[1, 2, 2, 2]
        self.num_classes = num_classes


    def size_2_structure(self):
        """

        :return:
        """
        choices = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
            200: [3, 24, 36, 3]
        }
        assert self.resnet_size in choices, \
            'ResNet size must be one of [18, 34, 50, 101, 152, 200], receive %d.' % self.resnet_size

        self.block_nums = choices[self.resnet_size]


    def forward(self):
        """

        :return:
        """
        self.size_2_structure()
        model = ResNet(self.resnet_size,
                       num_classes=self.num_classes,
                       start_channels=64,
                       block_nums=self.block_nums,
                       block_stride=self.block_stride,
                       is_classification=self.is_classification,
                       start_kernel_size=7,
                       start_conv_stride=2,
                       start_pool_size=3,
                       start_pool_stride=2,
                       data_format=self.data_format,
                       dtype=self.dtype)
        return model


def ResNet_forward(inPuts, resnet_size, is_classification=False, num_classes=None, is_train=False, data_format='channels_first'):
    """

    :param inPuts:
    :param resnet_size:
    :param is_classification:
    :param num_classes:
    :param data_format:
    :return:
    """
    return ResNet_use(resnet_size=resnet_size, is_classification=is_classification, num_classes=num_classes,
                      data_format=data_format).forward()(inPuts, is_train=is_train)

if __name__ == '__main__':
    model = ResNet_use(
        resnet_size=50,
        data_format='channels_first',
        is_classification=False,
        num_classes=None).forward()
    Test_Model(model, input_size=(1, 800, 800, 3), speed_test=Test_Model, print_tensor=True)

# if __name__ == '__main__':
#     inPuts = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
#     model = ResNet_use(resnet_size=50, is_classification=False, num_classes=None).forward()
#     output = model(inPuts, is_train=False)
#     variable_names = [v.name for v in tf.all_variables()]
#
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     data = np.random.randn(1, 800, 800, 3)
#     feed_dict = {inPuts: data}
#
#     Test_Speed(sess, feed_dict, output)

# if isinstance(o, list):
#     for o_ in o:
#         print(o_.shape)
# else:
#     print(o.shape)
#
# Print_Net(sess=sess)
# weight_path = 'D:\\Github\\General_Framework\\base\ResNet\\resnet_v2_fp32_savedmodel_NHWC\\1538687283\\variables\\variables.ckpt-01'
# saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_model'))
# saver.restore(sess, weight_path)
