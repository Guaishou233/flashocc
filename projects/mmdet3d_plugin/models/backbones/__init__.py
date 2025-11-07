from mmdet.models.backbones import ResNet
from .resnet import CustomResNet, CustomResNetV2
from .swin import SwinTransformer

__all__ = ['ResNet', 'CustomResNet', 'CustomResNetV2', 'SwinTransformer']
