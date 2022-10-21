# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""SSD Keras-based ResnetV1 Extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.models.keras_models import resnet_v1
from object_detection.utils import ops
from object_detection.utils import shape_utils

_RESNET_MODEL_OUTPUT_LAYERS = {
    'resnet_v1_34': ['conv2_block3_out', 'conv3_block4_out',
                     'conv4_block6_out', 'conv5_block3_out'],
    'resnet_v1_50': ['conv2_block3_out', 'conv3_block4_out',
                     'conv4_block6_out', 'conv5_block3_out'],
    'resnet_v1_101': ['conv2_block3_out', 'conv3_block4_out',
                      'conv4_block23_out', 'conv5_block3_out'],
    'resnet_v1_152': ['conv2_block3_out', 'conv3_block8_out',
                      'conv4_block36_out', 'conv5_block3_out'],
}

_RESNET_OUTPUT_LAYERS_DEPTH = {
    'resnet_v1_34': [256, 512, 512, 256, 256, 256],
    'resnet_v1_50': [1024, 512, 512, 256, 256, 256],
    'resnet_v1_101': [1024, 512, 512, 256, 256, 256],
    'resnet_v1_152': [1024, 512, 512, 256, 256, 256]
}



class SSDResNetV1KerasFeatureExtractor(
    ssd_meta_arch.SSDKerasFeatureExtractor):
  """SSD Feature Extractor using Keras-based ResnetV1 features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               resnet_v1_base_model,
               resnet_v1_base_model_name,
               inplace_batchnorm_update,
               use_explicit_padding=False,
               use_depthwise=False,
               num_layers=6,
               override_base_feature_extractor_hyperparams=False,
               name=None):
    """SSD Keras based FPN feature extractor Resnet v1 architecture.
    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      inplace_batchnorm_update: whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      resnet_v1_base_model: base resnet v1 network to use. One of
        the resnet_v1.resnet_v1_{34,50,101,152} models.
      resnet_v1_base_model_name: model name under which to construct resnet v1.
      fpn_min_level: the highest resolution feature map to use in FPN. The valid
        values are {2, 3, 4, 5} which map to Resnet blocks {1, 2, 3, 4}
        respectively.
      fpn_max_level: the smallest resolution feature map to construct or use in
        FPN. FPN constructions uses features maps starting from fpn_min_level
        upto the fpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of fpn
        levels.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: whether to reuse variables. Default is None.
      use_explicit_padding: whether to use explicit padding when extracting
        features. Default is None, as it's an invalid option and not implemented
        in this feature extractor.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams`.
      name: a string name scope to assign to the model. If 'None', Keras
        will auto-generate one from the class name.
    """
    super(SSDResNetV1KerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        use_explicit_padding=None,
        use_depthwise=None,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)
    if self._use_explicit_padding:
      raise ValueError('Explicit padding is not a valid option.')
    if self._use_depthwise:
      raise ValueError('Depthwise is not a valid option.')
    self._resnet_v1_base_model = resnet_v1_base_model
    self._resnet_v1_base_model_name = resnet_v1_base_model_name
    self.classification_backbone = None
    self._coarse_feature_layers = []
    
  def build(self, input_shape):
    if self._resnet_v1_base_model_name == 'resnet_v1_34':
        full_resnet_v1_model = self._resnet_v1_base_model(
            conv_hyperparams=(self._conv_hyperparams
                              if self._override_base_feature_extractor_hyperparams
                              else None),
            depth_multiplier=self._depth_multiplier,
            min_depth=self._min_depth,
            classes=None,
            weights=None,
            include_top=False)
    else:
        full_resnet_v1_model = self._resnet_v1_base_model(
            batchnorm_training=(self._is_training and not self._freeze_batchnorm),
            conv_hyperparams=(self._conv_hyperparams
                              if self._override_base_feature_extractor_hyperparams
                              else None),
            depth_multiplier=self._depth_multiplier,
            min_depth=self._min_depth,
            classes=None,
            weights=None,
            include_top=False)
    # full_resnet_v1_model.get_layer('conv3_block1_1_conv').strides=(1,1)
    # full_resnet_v1_model.get_layer('conv3_block1_0_conv').strides=(1,1)
    
    output_layers = _RESNET_MODEL_OUTPUT_LAYERS[self._resnet_v1_base_model_name]
    outputs = [full_resnet_v1_model.get_layer(output_layer_name).output
               for output_layer_name in output_layers]
    self.classification_backbone = tf.keras.Model(
        inputs=full_resnet_v1_model.inputs,
        outputs=outputs)
    
    # self._feature_map_layout = {
    #     'from_layer': ([output_layers[2]] + ['']*5)[:self._num_layers],
    #     'layer_depth': _RESNET_OUTPUT_LAYERS_DEPTH[self._resnet_v1_base_model_name][:self._num_layers],
    #     'use_explicit_padding': self._use_explicit_padding,
    #     'use_depthwise': self._use_depthwise,
    # }
    self._feature_map_layout = {
        'from_layer': (output_layers[1:3] + ['']*(6-len(output_layers[1:3])))[:self._num_layers],
        'layer_depth': _RESNET_OUTPUT_LAYERS_DEPTH[self._resnet_v1_base_model_name][:self._num_layers],
        'use_explicit_padding': self._use_explicit_padding,
        'use_depthwise': self._use_depthwise,
    }
    self.feature_map_generator = (
        feature_map_generators.KerasMultiResolutionFeatureMaps(
            feature_map_layout=self._feature_map_layout,
            depth_multiplier=self._depth_multiplier,
            min_depth=self._min_depth,
            insert_1x1_conv=True,
            is_training=self._is_training,
            conv_hyperparams=self._conv_hyperparams,
            freeze_batchnorm=self._freeze_batchnorm,
            name='FeatureMaps'))
    self.built = True

  def preprocess(self, resized_inputs):
    """SSD preprocessing.
    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-mdnge.
    Note that if the number of channels is not equal to 3, the mean subtraction
    will be skipped and the original resized_inputs will be returned.
    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (1.0 / 255.0) * resized_inputs

  def _extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.
    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        129, preprocessed_inputs)

    image_features = self.classification_backbone(
        ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple))
    output_layers = _RESNET_MODEL_OUTPUT_LAYERS[self._resnet_v1_base_model_name]
    # SKIP OUTPUT BLOCK 2nd
    feature_maps = self.feature_map_generator({output_layers[i]:image_features[i] for i in range(1,len(output_layers))})

    return list(feature_maps.values())


class SSDResNet50V1KerasFeatureExtractor(
    SSDResNetV1KerasFeatureExtractor):
  """SSD Feature Extractor using Keras-based ResnetV1-50 FPN features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               reuse_weights=None,
               use_explicit_padding=None,
               use_depthwise=None,
               override_base_feature_extractor_hyperparams=False,
               name='ResNet50V1'):
    """SSD Keras based FPN feature extractor ResnetV1-50 architecture.
    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      inplace_batchnorm_update: whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      fpn_min_level: the minimum level in feature pyramid networks.
      fpn_max_level: the maximum level in feature pyramid networks.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: whether to reuse variables. Default is None.
      use_explicit_padding: whether to use explicit padding when extracting
        features. Default is None, as it's an invalid option and not implemented
        in this feature extractor.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams`.
      name: a string name scope to assign to the model. If 'None', Keras
        will auto-generate one from the class name.
    """
    super(SSDResNet50V1KerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        resnet_v1_base_model=resnet_v1.resnet_v1_50,
        resnet_v1_base_model_name='resnet_v1_50',
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)

class SSDResNet34V1KerasFeatureExtractor(
    SSDResNetV1KerasFeatureExtractor):
  """SSD Feature Extractor using Keras-based ResnetV1-50 FPN features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               reuse_weights=None,
               use_explicit_padding=None,
               use_depthwise=None,
               override_base_feature_extractor_hyperparams=False,
               name='ResNet50V1'):
    """SSD Keras based FPN feature extractor ResnetV1-50 architecture.
    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      inplace_batchnorm_update: whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      fpn_min_level: the minimum level in feature pyramid networks.
      fpn_max_level: the maximum level in feature pyramid networks.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: whether to reuse variables. Default is None.
      use_explicit_padding: whether to use explicit padding when extracting
        features. Default is None, as it's an invalid option and not implemented
        in this feature extractor.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams`.
      name: a string name scope to assign to the model. If 'None', Keras
        will auto-generate one from the class name.
    """
    super(SSDResNet34V1KerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        resnet_v1_base_model=resnet_v1.resnet_v1_34,
        resnet_v1_base_model_name='resnet_v1_34',
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)
