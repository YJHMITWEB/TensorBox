from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import tensorflow as tf
from components.MobileNet import conv_blocks as ops
from components.MobileNet import mobilenet
from components.MobileNet import MobileNet_v2

slim = tf.contrib.slim

def find_ops(optype):
  """Find ops of a given type in graphdef or a graph.

  Args:
    optype: operation type (e.g. Conv2D)
  Returns:
     List of operations.
  """
  gd = tf.get_default_graph()
  return [var for var in gd.get_operations() if var.type == optype]

class MobileNet():
    def __init__(self, num_classes, is_classification, depth_multiply=1.0, data_format='channels_last'):
        """

        :param num_classes:
        :param is_train:
        :param depth_multiply:
        """
        self.num_classes = num_classes
        self.depth_multiply = depth_multiply
        self.is_classification = is_classification
        self.data_format = data_format

    def __call__(self, inPuts, is_train):
        """

        :param inPuts:
        :param is_train:
        :return:
        """
        net, end_points = MobileNet_v2.mobilenet(inPuts,
                                                 conv_defs=MobileNet_v2.V2_DEF,
                                                 depth_multiplier=self.depth_multiply,
                                                 num_classes=self.num_classes)
        return net, end_points