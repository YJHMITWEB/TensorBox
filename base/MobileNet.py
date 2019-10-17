import os
import sys
import tensorflow as tf
import numpy as np
sys.path.append(os.path.abspath(".."))
from models.MobileNet_v2 import MobileNet
from visualize.model_test import Test_Model
slim = tf.contrib.slim


class MobileNet_use():
    def __init__(self,
                 depth_multiply,
                 data_format='channels_last',
                 is_classification=False,
                 num_classes=None):
        """

        :param nasnet_large_or_small:
        :param data_format:
        :param dtype:
        :param is_classification:
        :param num_classes:
        """
        assert data_format in \
               ['channels_last', 'channels_first'], \
            'data_format must be either \'channels_last\' or \'channels_first\'.'
        self.depth_multiply = depth_multiply
        self.data_format = data_format
        self.is_classification = is_classification
        self.num_classes = num_classes

    def forward(self):
        """

        :return:
        """
        model = MobileNet(
            depth_multiply=self.depth_multiply,
            is_classification=self.is_classification,
            num_classes=self.num_classes,
            data_format=self.data_format)
        return model


if __name__ == '__main__':
    model = MobileNet_use(
        num_classes=None,
        is_classification=False,
        depth_multiply=1.0,
        data_format='channels_last').forward()
    Test_Model(model, input_size=(1, 800, 800, 3), speed_test=True, print_tensor=True)