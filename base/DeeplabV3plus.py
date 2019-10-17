import tensorflow as tf
import os
import sys
sys.path.append(os.path.abspath(".."))
from models.DeepLabv3plus import Deeplabv3plus
from components.DeepLab_V3plus import common
from visualize.model_test import Test_Model

class DeeplabV3plus_use():
    def __init__(self,
                 input_size=(800, 800),
                 output_stride=8,
                 num_classes=81,
                 model_mode='mobilenet_v2'):
        """

        :param input_size:
        :param output_stride:
        :param num_classes:
        :param model_mode: either ['xception_65', 'mobilenet_v2']
        """
        self.input_size = list(input_size)
        self.output_stride = output_stride
        self.num_classes = num_classes
        self.outputs_to_num_classes = {'semantic': num_classes}
        self.model_options = common.ModelOptions(
            self.outputs_to_num_classes,
            self.input_size,
            output_stride=self.output_stride
        )._replace(
            add_image_level_feature=True,
            aspp_with_batch_norm=True,
            logits_kernel_size=1,
            model_variant=model_mode)

    def forward(self):
        """

        :return:
        """
        model = Deeplabv3plus(
            input_size=self.input_size,
            output_stride=self.output_stride,
            model_options=self.model_options,)
        return model

if __name__ == '__main__':
    model = DeeplabV3plus_use(
        num_classes=81,
        output_stride=8,
        input_size=(8000, 8000),
        model_mode='xception_65').forward()
    Test_Model(model, input_size=(1, 8000, 8000, 3), speed_test=True, print_tensor=True)