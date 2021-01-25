# Data Transformation Guide

When you create **MultiTransformDataset()** object, you need pass a list transformation to deal with the different scenarios during the active learning loop.

- Parameter transforms (list of transformation) is list of callable transform object to be applie on a sample, using its index as the mode for the transformation. By default, 0 is the index of transformation used for training(with augmentation), 1 is the index of transformation used for predicting(without augmentation).
- If `MEBaseModel` is used, `TransformTwice` object need to be appended to the `tranformations` list as the last item. The input for TransformTwice wrapper should be the transformation used for training, that is, identical to the first transformation in the list.
- For the NoisyStudent Algorithm, we recommend to add transformation of Brightness, Contrast, Sharpness during the training step to help student model outperform their teacher. For more information, refer to [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf)

