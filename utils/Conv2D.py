import tensorflow as tf
from .Padding import Padding


def Conv2D(inPuts, out_channels, kernel_size, stride, data_format='NHWC'):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if stride > 1:
    inPuts = Padding(inPuts, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inPuts, filters=out_channels, kernel_size=kernel_size, strides=stride,
      padding=('SAME' if stride == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)
