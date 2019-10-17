import os
import sys
import tensorflow as tf
import numpy as np
sys.path.append(os.path.abspath(".."))
from models.VGG import VGG
from visualize.model_test import Test_Model
slim = tf.contrib.slim


class VGG_use():
    def __init__(self, layer_nums, num_classes, data_format, name='vgg_19'):
        """

        :param layer_nums:
        :param num_classes:
        :param data_format:
        :param name:
        """
        self.layer_name = layer_nums
        self.num_classes = num_classes
        self.data_format = data_format
        self.name = name

    def forward(self):
        """

        :param inPuts:
        :param is_train:
        :return:
        """
        model = VGG(layer_nums=self.layer_name, num_classes=self.num_classes, data_format=self.data_format)
        return model

if __name__ == '__main__':
    model = VGG_use(
        layer_nums=16,
        num_classes=None,
        data_format='channels_last').forward()
    Test_Model(model, input_size=(1, 800, 800, 3), speed_test=True, print_tensor=True)