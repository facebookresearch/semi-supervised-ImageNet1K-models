# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Optional list of dependencies required by the package
dependencies = ['torch', 'torchvision']

from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck
import torchvision.models as models

semi_supervised_model_urls = {
    'resnet18':         'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth',
    'resnet50':         'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth',    
    'resnext50_32x4d':  'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth',
    'resnext101_32x4d': 'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth',
    'resnext101_32x8d': 'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth',
    'resnext101_32x16d':'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth',
}

semi_weakly_supervised_model_urls = {
    'resnet18':         'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth',
    'resnet50':         'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth',    
    'resnext50_32x4d':  'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth',
    'resnext101_32x4d': 'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth',
    'resnext101_32x8d': 'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth',
    'resnext101_32x16d':'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth',
}


def _resnext(url, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    state_dict = load_state_dict_from_url(url, progress=progress)
    model.load_state_dict(state_dict)
    return model


def _resnet(url, depth, pretrained, progress, **kwargs):
    if depth == 50:
        model = models.resnet50(pretrained=pretrained, **kwargs)
    elif depth == 18:
        model = models.resnet18(pretrained=pretrained, **kwargs)
    else:
        print('ERROR: only ResNet-18 and ResNet-50 models are available.')
    state_dict = load_state_dict_from_url(url, progress=progress)
    model.load_state_dict(state_dict)
    return model


# -------------- semi-supervised models ----------------
def resnet18_ssl(progress=True, **kwargs):
    """Constructs a semi-supervised ResNet-18 model pre-trained on YFCC100M dataset and finetuned on ImageNet 
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    return _resnet(semi_supervised_model_urls['resnet18'], 18, True, progress, **kwargs)


def resnet50_ssl(progress=True, **kwargs):
    """Constructs a semi-supervised ResNet-50 model pre-trained on YFCC100M dataset and finetuned on ImageNet 
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(semi_supervised_model_urls['resnet50'], 50, True, progress, **kwargs)


def resnext50_32x4d_ssl(progress=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-50 32x4 model pre-trained on YFCC100M dataset and finetuned on ImageNet 
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnext(semi_supervised_model_urls['resnext50_32x4d'], Bottleneck, [3, 4, 6, 3], True, progress, **kwargs)


def resnext101_32x4d_ssl(progress=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-101 32x4 model pre-trained on YFCC100M dataset and finetuned on ImageNet 
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnext(semi_supervised_model_urls['resnext101_32x4d'], Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


def resnext101_32x8d_ssl(progress=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-101 32x8 model pre-trained on YFCC100M dataset and finetuned on ImageNet 
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext(semi_supervised_model_urls['resnext101_32x8d'], Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


def resnext101_32x16d_ssl(progress=True, **kwargs):
    """Constructs a semi-supervised ResNeXt-101 32x16 model pre-trained on YFCC100M dataset and finetuned on ImageNet 
    `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    return _resnext(semi_supervised_model_urls['resnext101_32x16d'], Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


# -------------- semi-weakly supervised models ----------------
def resnet18_swsl(progress=True, **kwargs):
    """Constructs a semi-weakly supervised Resnet-18 model pre-trained on 1B weakly supervised 
       image dataset and finetuned on ImageNet.  
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    return _resnet(semi_weakly_supervised_model_urls['resnet18'], 18, True, progress, **kwargs)


def resnet50_swsl(progress=True, **kwargs):
    """Constructs a semi-weakly supervised ResNet-50 model pre-trained on 1B weakly supervised 
       image dataset and finetuned on ImageNet.  
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(semi_weakly_supervised_model_urls['resnet50'], 50, True, progress, **kwargs)


def resnext50_32x4d_swsl(progress=True, **kwargs):
    """Constructs a semi-weakly supervised ResNeXt-50 32x4 model pre-trained on 1B weakly supervised 
       image dataset and finetuned on ImageNet.  
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnext(semi_weakly_supervised_model_urls['resnext50_32x4d'], Bottleneck, [3, 4, 6, 3], True, progress, **kwargs)


def resnext101_32x4d_swsl(progress=True, **kwargs):
    """Constructs a semi-weakly supervised ResNeXt-101 32x4 model pre-trained on 1B weakly supervised 
       image dataset and finetuned on ImageNet.  
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnext(semi_weakly_supervised_model_urls['resnext101_32x4d'], Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


def resnext101_32x8d_swsl(progress=True, **kwargs):
    """Constructs a semi-weakly supervised ResNeXt-101 32x8 model pre-trained on 1B weakly supervised 
       image dataset and finetuned on ImageNet.  
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext(semi_weakly_supervised_model_urls['resnext101_32x8d'], Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


def resnext101_32x16d_swsl(progress=True, **kwargs):
    """Constructs a semi-weakly supervised ResNeXt-101 32x16 model pre-trained on 1B weakly supervised 
       image dataset and finetuned on ImageNet.  
       `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    return _resnext(semi_weakly_supervised_model_urls['resnext101_32x16d'], Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)
