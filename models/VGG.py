from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import tensorflow as tf
from components.VGG import vgg

slim = tf.contrib.slim

class VGG():
    def __init__(self, layer_nums, num_classes, data_format='channels_last'):
        """

        :param layer_nums:
        :param num_classes:
        :param data_format:
        """
        assert layer_nums in [11, 16, 19], 'VGG should has one of [11, 16, 19] layers.'
        self.layer_nums = layer_nums
        self.num_classes = num_classes
        self.data_format = data_format


    def __call__(self, inPuts, is_train):
        """

        :param inPuts:
        :param is_train:
        :return:
        """
        if self.layer_nums == 11:
            net, end_points = vgg.vgg_a(inPuts, num_classes=self.num_classes, is_training=is_train, scope='vgg_a')
        elif self.layer_nums == 16:
            net, end_points = vgg.vgg_16(inPuts, num_classes=self.num_classes, is_training=is_train, scope='vgg_16')
        else:
            net, end_points = vgg.vgg_19(inPuts, num_classes=self.num_classes, is_training=is_train, scope='vgg_19')

        return net, end_points