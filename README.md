## Semi-Supervised and Semi-Weakly Supervised ImageNet Models

This project includes the semi-supervised and semi-weakly supervised ImageNet models introduced in "Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>. 

"Semi-supervised" (SSL) ImageNet models are pre-trained on a subset of unlabeled YFCC100M public image dataset and fine-tuned with the ImageNet1K training dataset, as described by the semi-supervised training framework in the paper mentioned above. In this case, the high capacity teacher model was trained only with labeled examples. 

"Semi-weakly" supervised (SWSL) ImageNet models are pre-trained on **940 million** public images with 1.5K hashtags matching with 1000 ImageNet1K synsets, followed by fine-tuning on ImageNet1K dataset. In this case, the associated hashtags are only used for building a better teacher model. During training the student model, those hashtags are ingored and the student model is pretrained with a subset of 64M images selected by the teacher model from the same 940 million public image dataset. 

We are providing the following semi-supervised and semi-weakly supervised ImageNet models. The teacher models used for training these models have the ResNet-101-32x48 model architecture. 

Semi-weakly supervised ResNet and ResNext models provided in the table below significantly improve the top-1 accuracy on the ImageNet validation set compared to training from scratch or other training mechanisms introduced in the literature as of September 2019. For example, **We achieve state-of-the-art accuracy of 81.2% on ImageNet for the widely used/adopted ResNet-50 model architecture**. 


| Architecture       |   Supervision   | #Parameters | FLOPS | Top-1 Acc. | Top-5 Acc. |
| ------------------ | :--------------:|:----------: | :---: | :--------: | :--------: |
| ResNet-18          | semi-supervised        |14M     | 2B   |     72.8      | 91.5    |
| ResNet-50          | semi-supervised        |25M     | 4B   |     79.3      | 94.9    |    
| ResNeXt-50 32x4d   | semi-supervised        |25M     | 4B   |     80.3      | 95.4    |
| ResNeXt-101 32x4d  | semi-supervised        |42M     | 8B   |     81.0      | 95.7    |
| ResNeXt-101 32x8d  | semi-supervised        |88M     | 16B   |     81.7    |  96.1   |
| ResNeXt-101 32x16d | semi-supervised        |193M    | 36B   |     81.9   | 96.2     |
| ResNet-18          | semi-weakly supervised |14M     | 2B   |    **73.4**    |  91.9      |
| ResNet-50          | semi-weakly supervised |25M     | 4B   |    **81.2**    |  96.0      |     
| ResNeXt-50 32x4d   | semi-weakly supervised |25M     | 4B   |    **82.2**    |  96.3      |
| ResNeXt-101 32x4d  | semi-weakly supervised |42M     | 8B   |    **83.4**    |  96.8      |
| ResNeXt-101 32x8d  | semi-weakly supervised |88M     | 16B   |  **84.3**    |  97.2    |
| ResNeXt-101 32x16d | semi-weakly supervised |193M    | 36B   |  **84.8**    |  97.4    |


## Loading models with torch.hub
The models are available with [torch.hub](https://pytorch.org/docs/stable/hub.html).
As an example, to load the semi-weakly trained ResNet-50 model, simply run:

```
model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
```
Please refer to [torch.hub](https://pytorch.org/docs/stable/hub.html) to see a full example of using the model to classify an image.

## Citation

If you use the models released in this repository, please cite the following publication.
```
@misc{yalniz2019billionscale,
    title={Billion-scale semi-supervised learning for image classification},
    author={I. Zeki Yalniz and Hervé Jégou and Kan Chen and Manohar Paluri and Dhruv Mahajan},
    year={2019},
    eprint={1905.00546},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## License
The models in this repository are released under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for additional details.
