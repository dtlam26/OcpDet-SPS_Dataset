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

class SSDVggKerasFeatureExtractor(
    ssd_meta_arch.SSDKerasFeatureExtractor):
  """SSD Feature Extractor using Keras-based Vggfeatures."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               default_shape=300,
               use_explicit_padding=False,
               use_depthwise=False,
               num_layers=6,
               override_base_feature_extractor_hyperparams=False,
               name=None):
    """SSD Keras based FPN feature extractor VGG 300 input shape architecture.
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
    super(SSDVggKerasFeatureExtractor, self).__init__(
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
    
    self.default_shape = default_shape
    if self._use_explicit_padding:
      raise ValueError('Explicit padding is not a valid option.')
    if self._use_depthwise:
      raise ValueError('Depthwise is not a valid option.')
    self.classification_backbone = None
    self._coarse_feature_layers = []
    
  def build(self, input_shape):

    full_vgg_model = tf.keras.applications.VGG16(
                        include_top=False,
                        weights="imagenet",
                        input_tensor=None,
                        input_shape=None,
                        pooling=None,
                        classes=1000,
                    )
    # print("Frozen Backbone")
    # for i in range(len(full_vgg_model.layers)):
    #     full_vgg_model.layers[i].trainable = False
        
    conv6 = tf.keras.layers.Conv2D(1024, 3, strides=(1, 1), padding='same', activation='relu', dilation_rate=(6,6), name='conv6')(full_vgg_model.layers[-1].output)
    conv7 = tf.keras.layers.Conv2D(1024, 1, strides=(1, 1), padding='same', activation='relu', name='conv7')(conv6)
    block4_conv3 = full_vgg_model.get_layer('block4_conv3').output
    self.classification_backbone = tf.keras.Model(
        inputs=full_vgg_model.inputs,
        outputs=[block4_conv3,conv7])
    
    self.output_layers = ['block4_conv3','conv7']
    
    if self.default_shape == 300:
        self._feature_map_layout = {
        'from_layer': ['block4_conv3','conv7','','','',''][:self._num_layers],
        'layer_depth': [-1,-1,512,256,256,256],
        'use_explicit_padding': self._use_explicit_padding,
        'use_depthwise': self._use_depthwise,
    }
        
    else:
        self._feature_map_layout = {
        'from_layer': ['','conv7','','','',''][:self._num_layers],
        'corresponding_base': ['block4_conv3','','','','',''],
        'layer_depth': [1024,-1,512,256,256,256],
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
            skip_bn=True,
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
    
    # SKIP OUTPUT BLOCK 2nd
    feature_maps = self.feature_map_generator({self.output_layers[i]:image_features[i] for i in range(len(self.output_layers))})

    return list(feature_maps.values())

class SSDVgg512KerasFeatureExtractor(
    SSDVggKerasFeatureExtractor):

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
               name='VGG_512'):

    super(SSDVgg512KerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        default_shape=512,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)
    
class SSDVgg300KerasFeatureExtractor(
    SSDVggKerasFeatureExtractor):
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
               name='VGG_300'):
    
    super(SSDVgg300KerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        default_shape=300,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)