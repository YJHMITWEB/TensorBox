import os
import sys
import tensorflow as tf
import numpy as np
sys.path.append(os.path.abspath(".."))
from models.NasNet import NasNet
from visualize.model_test import Test_Model
slim = tf.contrib.slim


class NasNet_use():
    def __init__(self,
                 nasnet_large_or_small,
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
        self.nasnet_large_or_small = nasnet_large_or_small
        self.data_format = data_format
        self.is_classification = is_classification
        self.num_classes = num_classes

    def forward(self):
        """

        :return:
        """
        model = NasNet(
            nasnet_large_or_small=self.nasnet_large_or_small,
            is_classification=self.is_classification,
            num_classes=self.num_classes,
            data_format=self.data_format)
        return model

def NasNet_forward(inPuts, nasnet_large_or_small='small', is_classification=False,
                   num_classes=None, is_train=False, data_format='channels_last'):
    """

    :param inPuts:
    :param nasnet_large_or_small:
    :param is_classification:
    :param num_classes:
    :param is_train:
    :param data_format:
    :return:
    """
    return NasNet_use(nasnet_large_or_small=nasnet_large_or_small, is_classification=is_classification,
                      num_classes=num_classes, data_format=data_format).forward()(inPuts, is_train=is_train)

if __name__ == '__main__':
    model = NasNet_use(
        nasnet_large_or_small='small',
        data_format='channels_first',
        is_classification=False,
        num_classes=None).forward()
    Test_Model(model, input_size=(1, 800, 800, 3), speed_test=True, print_tensor=True)