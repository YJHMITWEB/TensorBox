"""
The same as in components/Xception/Xception, because the official implementation of Xception uses tf.contrib.slim,
which is dangerous to simply convert into tf open source codes.
"""
from components.Xception.Xception import xception_41, xception_65, xception_71, xception_arg_scope
import tensorflow as tf
slim = tf.contrib.slim
class Xception():
    def __init__(self,
                 layer_nums,
                 is_classification=False,
                 num_classes=None,
                 global_pool=True,
                 keep_prob=0.5,
                 output_stride=None,
                 regularize_depthwise=False,
                 multi_grid=None,
                 reuse=None):
        """

        :param layer_nums:
        """
        assert layer_nums in [41, 65, 71], 'Layer_nums must be one of [41, 65, 71].'
        self.layer_nums = layer_nums
        self.is_classification = is_classification
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.keep_prob = keep_prob
        self.output_stride = output_stride
        self.regularize_depthwise = regularize_depthwise
        self.multi_grid = multi_grid
        self.reuse = reuse

    def __call__(self, inPuts, is_train):
        with slim.arg_scope(xception_arg_scope()):
            if self.layer_nums == 41:
                net, endpoints = xception_41(inPuts,
                        num_classes=self.num_classes,
                        is_training=is_train,
                        is_classification=self.is_classification,
                        global_pool=self.global_pool,
                        keep_prob=self.keep_prob,
                        output_stride=self.output_stride,
                        regularize_depthwise=self.regularize_depthwise,
                        multi_grid=self.multi_grid,
                        reuse=self.reuse,
                        scope='xception_41')

            if self.layer_nums == 65:
                net, endpoints = xception_65(inPuts,
                        num_classes=self.num_classes,
                        is_training=is_train,
                        is_classification=self.is_classification,
                        global_pool=self.global_pool,
                        keep_prob=self.keep_prob,
                        output_stride=self.output_stride,
                        regularize_depthwise=self.regularize_depthwise,
                        multi_grid=self.multi_grid,
                        reuse=self.reuse,
                        scope='xception_65')

            if self.layer_nums == 71:
                net, endpoints = xception_71(inPuts,
                        num_classes=self.num_classes,
                        is_training=is_train,
                        is_classification=self.is_classification,
                        global_pool=self.global_pool,
                        keep_prob=self.keep_prob,
                        output_stride=self.output_stride,
                        regularize_depthwise=self.regularize_depthwise,
                        multi_grid=self.multi_grid,
                        reuse=self.reuse,
                        scope='xception_71')

            return endpoints