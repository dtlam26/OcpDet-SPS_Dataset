#! /bin/bash

if [ $1 ]; then
    object_detpath=`realpath $1`
else
    object_detpath="/home/lam/anaconda3/envs/tf-2.x/lib/python3.9/site-packages/object_detection"
fi

echo "Override object detection module: ${object_detpath}"

#exporter
cp exporter_lib_v2.py "${object_detpath}"

#models
cp feature_map_generators.py "${object_detpath}/models"
cp ssd_mobilenet_v2_keras_learning_loss_feature_extractor.py "${object_detpath}/models"
cp ssd_mobilenet_fpn_keras_learning_loss_feature_extractor.py "${object_detpath}/models"
cp ssd_mobilenet_fpn_keras_learning_loss_active_anchors_feature_extractor.py "${object_detpath}/models"
cp ssd_resnet_feature_extractor.py "${object_detpath}/models"
cp ssd_vgg_feature_extractor.py "${object_detpath}/models"

#meta arch
cp ssd_meta_arch.py "${object_detpath}/meta_architectures"

#losses
cp losses.py "${object_detpath}/core"

#builders
cp model_builder.py "${object_detpath}/builders"
cp losses_builder.py "${object_detpath}/builders"

#box_predictors
cp convolutional_keras_box_predictor.py "${object_detpath}/predictors"
cp keras_box_head.py "${object_detpath}/predictors/heads"
cp keras_class_head.py "${object_detpath}/predictors/heads"
cp head.py "${object_detpath}/predictors/heads"

echo "$(date)"