# TensorBox
*A user-friendly API for advanced computer vision models*

TensorBox provides well-build models in recognition and segmentation. All have been tested on corresponding datasets, such as ImageNet, Pascal VOC, Microsoft COCO, etc. To call well-built models and train/test, user only need to import its function, and writing less than 10 rows of code to get it running.

Also, basic modules such as residual block, dilation convs block, reduction cell are available for direct calling, in case user wants to build their own networks. Note that, to make the model's file more stractified, each category level contains very small amount of code which mainly calls lower level's functions.

Visualization tools are provided, so that user could call the tools on any models they build to test the processing speed as well as memory consumption.

At this point, the models that already built-in are 
### Recognition:
+ MobileNet
+ VGG
+ NasNet
+ Xception
+ ResNet
### Semantic Segmentation:
+ DeepLab v3+

All models are tested both in terms of speed and accuracy. Entries we provide could almost reach the fastest training/inference speed, results on a single Geforce TITAN X(Pascal) are listed below.
All pre-trained weights file will be available soon.
