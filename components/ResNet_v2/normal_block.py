import sys
import os
sys.path.append(os.path.abspath("../../utils"))

from utils.Conv2D import Conv2D
from utils.BatchNorm import BatchNorm
from utils.ReLU import ReLU
from utils.Add import Add

def Normal_Block(inPuts, out_channels, is_train, projection_shortcut, stride, data_format='NHWC'):
    """

    :param inputs:
    :param out_channels:
    :param is_train:
    :param shortcut_methods:
    :param stride:
    :param data_format:
    :return:
    """
    shortCut = inPuts
    inPuts = BatchNorm(inPuts=inPuts, is_train=is_train, data_format=data_format)
    inPuts = ReLU(inPuts)

    if projection_shortcut is True:
        shortCut = Conv2D(inPuts=shortCut, out_channels=out_channels, kernel_size=1, stride=stride, data_format=data_format)

    inPuts = Conv2D(inPuts=inPuts, out_channels=out_channels, kernel_size=3, stride=stride, data_format=data_format)
    inPuts = ReLU(inPuts)
    inPuts = Conv2D(inPuts=inPuts, out_channels=out_channels, kernel_size=3, stride=1, data_format=data_format)

    outPuts = Add(inPuts_A=inPuts, inPuts_B=shortCut)
    return outPuts