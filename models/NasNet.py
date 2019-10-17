from components.NasNet.NasNet import build_nasnet_large, build_nasnet_mobile, nasnet_mobile_arg_scope, nasnet_large_arg_scope
import tensorflow as tf
slim = tf.contrib.slim

class NasNet():
    def __init__(self,
                 nasnet_large_or_small='small',
                 is_classification=False,
                 num_classes=None,
                 data_format='channels_first'):
        """

        :param nasnet_large:
        :param nasnet_mobile:
        :param is_classification:
        """
        assert nasnet_large_or_small in ['small', 'large'], 'NasNet_large_or_small must be either \'large\' or \'small\'.'
        self.nasnet_large_or_small = nasnet_large_or_small
        self.is_classification = is_classification
        self.num_classes = num_classes
        self.data_format = data_format


    def __call__(self, inPuts, is_train):
        if self.nasnet_large_or_small == 'small':
            with slim.arg_scope(nasnet_mobile_arg_scope()):
                _, end_points = build_nasnet_mobile(inPuts, self.num_classes, is_training=is_train, data_format=self.data_format)
        else:
            with slim.arg_scope(nasnet_large_arg_scope()):
                _, end_points = build_nasnet_large(inPuts, self.num_classes, is_training=is_train, data_format=self.data_format)

        return end_points