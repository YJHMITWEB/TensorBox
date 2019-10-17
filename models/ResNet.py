import sys
import os
import tensorflow as tf
from components.ResNet_v2.bottleneck_block import Bottleneck_Block
from components.ResNet_v2.normal_block import Normal_Block
from components.ResNet_v2.block_groups import Block_Groups
from utils.Dtype import Float16, Float32
from utils.Conv2D import Conv2D
from utils.Identity import Identity
from utils.BatchNorm import BatchNorm
from utils.Transpose import Transpose
from utils.MaxPool2D import MaxPool2D
from utils.ReLU import ReLU
from utils.Mean import Mean
from utils.Remove_axis import Remove_axis
from utils.FullyConnect import FullyConnect


class ResNet():
    def __init__(self,
                 layer_nums,
                 num_classes,
                 start_channels,
                 block_nums,
                 block_stride,
                 is_classification=False,
                 start_kernel_size=3,
                 start_conv_stride=2,
                 start_pool_stride=2,
                 start_pool_size=2,
                 data_format=None,
                 dtype='float32'):
        """

        :param layer_nums:
        :param is_bottleneck:
        :param num_classes:
        :param start_channels:
        :param block_nums:
        :param block_strides:
        :param kernel_size:
        :param start_stride:
        :param start_pool_size:
        :param data_format:
        :param dtype:
        """

        if not data_format:
            data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        self.is_bottleneck = layer_nums >= 50
        self.data_format = data_format
        self.num_classes = num_classes
        self.start_channels = start_channels
        self.block_nums = block_nums
        self.block_stride = block_stride
        self.start_kernel_size = start_kernel_size
        self.start_conv_stride = start_conv_stride
        self.start_pool_stride = start_pool_stride
        self.start_pool_size = start_pool_size
        self.is_classification = is_classification
        self.dtype = Float32() if dtype == 'float32' else Float16()

    def _custom_dtype_getter(self, getter, name, shape=None, dtype=Float32(),
                             *args, **kwargs):
        """Creates variables in fp32, then casts to fp16 if necessary.

        This function is a custom getter. A custom getter is a function with the
        same signature as tf.get_variable, except it has an additional getter
        parameter. Custom getters can be passed as the `custom_getter` parameter of
        tf.variable_scope. Then, tf.get_variable will call the custom getter,
        instead of directly getting a variable itself. This can be used to change
        the types of variables that are retrieved with tf.get_variable.
        The `getter` parameter is the underlying variable getter, that would have
        been called if no custom getter was used. Custom getters typically get a
        variable with `getter`, then modify it in some way.

        This custom getter will create an fp32 variable. If a low precision
        (e.g. float16) variable was requested it will then cast the variable to the
        requested dtype. The reason we do not directly create variables in low
        precision dtypes is that applying small gradients to such variables may
        cause the variable not to change.

        Args:
          getter: The underlying variable getter, that has the same signature as
            tf.get_variable and returns a variable.
          name: The name of the variable to get.
          shape: The shape of the variable to get.
          dtype: The dtype of the variable to get. Note that if this is a low
            precision dtype, the variable will be created as a tf.float32 variable,
            then cast to the appropriate dtype
          *args: Additional arguments to pass unmodified to getter.
          **kwargs: Additional keyword arguments to pass unmodified to getter.

        Returns:
          A variable which is cast to fp16 if necessary.
        """

        return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self):
        """Returns a variable scope that the model should be created under.

        If self.dtype is a castable type, model variable will be created in fp32
        then cast to self.dtype before being used.

        Returns:
          A variable scope for the model.
        """

        return tf.variable_scope('resnet_model',
                                 custom_getter=self._custom_dtype_getter)


    def __call__(self, inPuts, is_train):
        """

        :param inputs: Default the inputs is "NHWC"
        :param is_train:
        :return:
        """
        with self._model_variable_scope():
            # Outter name scope is 'resnet_model'
            if self.data_format == 'channels_first':
                # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
                # This provides a large performance boost on GPU. See
                # https://www.tensorflow.org/performance/performance_guide#data_formats
                inPuts = Transpose(inPuts=inPuts, axis=(0, 3, 1, 2))

            inPuts = Conv2D(inPuts=inPuts,
                            out_channels=self.start_channels,
                            kernel_size=self.start_kernel_size,
                            stride=self.start_conv_stride,
                            data_format=self.data_format)
            inPuts = Identity(inPuts=inPuts, name='initial_conv')

            if self.start_pool_size:
                inPuts = MaxPool2D(inPuts=inPuts,
                                   kernel_size=self.start_pool_size,
                                   stride=self.start_pool_stride,
                                   padding='s',
                                   data_format=self.data_format)
                inPuts = Identity(inPuts, name='initial_max_pool')

            if self.is_classification is False:
                end_points = []
            for i, block_num in enumerate(self.block_nums):
                this_stage_channels = self.start_channels * (2 ** i)
                inPuts = Block_Groups(inPuts=inPuts,
                                      out_channels=this_stage_channels,
                                      is_bottleneck=self.is_bottleneck,
                                      block_nums=block_num,
                                      stride=self.block_stride[i],
                                      is_train=is_train,
                                      name='block_layer{}'.format(i + 1),
                                      data_format=self.data_format)
                if self.is_classification is False:
                    end_points.append(inPuts)
            inPuts = BatchNorm(inPuts, is_train=is_train, data_format=self.data_format)
            inPuts = ReLU(inPuts)

            if self.is_classification is True:
                assert self.num_classes is not None, 'Num_classes must be passed in.'
                axises = [2, 3] if self.data_format == 'channels_first' else [1, 2]
                inPuts = Mean(inPuts, axis=tuple(axises), keep_dim=True)
                inPuts = Identity(inPuts, 'final_reduce_mean')

                inPuts = Remove_axis(inPuts, axis=tuple(axises))
                inPuts = FullyConnect(inPuts, units=self.num_classes, is_train=is_train)
                inPuts = Identity(inPuts, name='final_dense')
                return inPuts
            else:
                return end_points