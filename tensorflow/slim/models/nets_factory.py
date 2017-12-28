# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf


from models import inception
from models import resnet_v1
from models import resnet_v2
from models import vgg

slim = tf.contrib.slim

networks_map = {
                'inception_v1': inception.inception_v1,
                'inception_v2': inception.inception_v2,
                'inception_v3': inception.inception_v3,
                'inception_v4': inception.inception_v4,
                'inception_resnet_v2': inception.inception_resnet_v2,
                'resnet_v2_50': resnet_v2.resnet_v2_50,
                'resnet_v2_101': resnet_v2.resnet_v2_101,
                'resnet_v2_152': resnet_v2.resnet_v2_152,
                'resnet_v1_50': resnet_v1.resnet_v1_50,
                'resnet_v1_101': resnet_v1.resnet_v1_101,
                'resnet_v1_152': resnet_v1.resnet_v1_152,                
                'vgg_a': vgg.vgg_a,
                'vgg_16': vgg.vgg_16,
                'vgg_19': vgg.vgg_19              
               }

arg_scopes_map = {
                  'inception_v1': inception.inception_v1_arg_scope,
                  'inception_v2': inception.inception_v2_arg_scope,
                  'inception_v3': inception.inception_v3_arg_scope,
                  'inception_v4': inception.inception_v4_arg_scope,
                  'inception_resnet_v2': inception.inception_resnet_v2_arg_scope,
                  'resnet_v2_50': resnet_v2.resnet_arg_scope,
                  'resnet_v2_101': resnet_v2.resnet_arg_scope,
                  'resnet_v2_152': resnet_v2.resnet_arg_scope,
                  'resnet_v1_50': resnet_v1.resnet_arg_scope,
                  'resnet_v1_101': resnet_v1.resnet_arg_scope,
                  'resnet_v1_152': resnet_v1.resnet_arg_scope,
                  'vgg_a': vgg.vgg_arg_scope,
                  'vgg_16': vgg.vgg_arg_scope,
                  'vgg_19': vgg.vgg_arg_scope                  
                 }


def get_inception_vgg_network_fn(name, num_classes, weight_decay=0.0, dropout_keep_prob=0.8,is_training=False, reuse=None):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        logits, end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
  """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  func = networks_map[name]
  @functools.wraps(func)
  def network_fn(images):
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
      return func(images, num_classes, is_training=is_training, dropout_keep_prob=dropout_keep_prob, reuse=reuse)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn

def get_res_network_fn(name, num_classes, weight_decay=0.0, is_training=False, reuse=None):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        logits, end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
   """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  func = networks_map[name]
  @functools.wraps(func)
  def network_fn(images):
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
      return func(images, num_classes, is_training=is_training, reuse=reuse)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn
