import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_utils
from .Padding import Dilation_Padding
from .Transpose import Transpose

slim = tf.contrib.slim

def separable_conv2d_same(inputs,
                          num_outputs,
                          kernel_size,
                          depth_multiplier,
                          stride,
                          rate=1,
                          use_explicit_padding=True,
                          regularize_depthwise=False,
                          scope=None,
                          data_format='channels_first',
                          **kwargs):
  """Strided 2-D separable convolution with 'SAME' padding.

  If stride > 1 and use_explicit_padding is True, then we do explicit zero-
  padding, followed by conv2d with 'VALID' padding.

  Note that

     net = separable_conv2d_same(inputs, num_outputs, 3,
       depth_multiplier=1, stride=stride)

  is equivalent to

     net = slim.separable_conv2d(inputs, num_outputs, 3,
       depth_multiplier=1, stride=1, padding='SAME')
     net = resnet_utils.subsample(net, factor=stride)

  whereas

     net = slim.separable_conv2d(inputs, num_outputs, 3, stride=stride,
       depth_multiplier=1, padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function.

  Consequently, if the input feature map has even height or width, setting
  `use_explicit_padding=False` will result in feature misalignment by one pixel
  along the corresponding dimension.

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    use_explicit_padding: If True, use explicit padding to make the model fully
      compatible with the open source version, otherwise use the native
      Tensorflow 'SAME' padding.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    scope: Scope.
    **kwargs: additional keyword arguments to pass to slim.conv2d

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """

  if data_format == 'channels_first':
      inputs = Transpose(inputs, (0, 2, 3, 1))

  def _separable_conv2d(padding):
    """Wrapper for separable conv2d."""
    return slim.separable_conv2d(inputs,
                                 num_outputs,
                                 kernel_size,
                                 depth_multiplier=depth_multiplier,
                                 stride=stride,
                                 rate=rate,
                                 padding=padding,
                                 scope=scope,
                                 **kwargs)
  def _split_separable_conv2d(padding):
    """Splits separable conv2d into depthwise and pointwise conv2d."""
    outputs = slim.separable_conv2d(inputs,
                                    None,
                                    kernel_size,
                                    depth_multiplier=depth_multiplier,
                                    stride=stride,
                                    rate=rate,
                                    padding=padding,
                                    scope=scope + '_depthwise',
                                    **kwargs)
    return slim.conv2d(outputs,
                       num_outputs,
                       1,
                       scope=scope + '_pointwise',
                       **kwargs)
  if stride == 1 or not use_explicit_padding:
    if regularize_depthwise:
      outputs = _separable_conv2d(padding='SAME')
    else:
      outputs = _split_separable_conv2d(padding='SAME')
  else:
    inputs = Dilation_Padding(inputs, kernel_size, rate, data_format=data_format)
    if regularize_depthwise:
      outputs = _separable_conv2d(padding='VALID')
    else:
      outputs = _split_separable_conv2d(padding='VALID')
  return outputs if data_format == 'channels_last' else Transpose(outputs, (0, 3, 1, 2))
