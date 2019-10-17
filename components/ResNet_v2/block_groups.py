from .bottleneck_block import Bottleneck_Block
from .normal_block import Normal_Block
import sys
import os
sys.path.append(os.path.abspath("../../utils"))
from utils.Identity import Identity

def Block_Groups(inPuts, out_channels, is_bottleneck, block_nums, stride, is_train, name, data_format):
    """

    :param inPuts:
    :param out_channels:
    :param is_bottleneck:
    :param block_nums:
    :param stride:
    :param is_train:
    :param name:
    :param data_format:
    :return:
    """
    block_fn = Bottleneck_Block if is_bottleneck else Normal_Block

    inPuts = block_fn(
        inPuts=inPuts,
        out_channels=out_channels,
        is_train=is_train,
        projection_shortcut=True,
        stride=stride,
        data_format=data_format)

    for i in range(1, block_nums):
        inPuts = block_fn(inPuts=inPuts,
                        out_channels=out_channels,
                        is_train=is_train,
                        projection_shortcut=False,
                        stride=1,
                        data_format=data_format)

    return Identity(inPuts=inPuts, name=name)