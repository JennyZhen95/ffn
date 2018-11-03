# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simplest FFN model, as described in https://arxiv.org/abs/1611.00421."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .. import model
from . import convstack_3d
from .. import optimizer
from tensorflow.python.util import deprecation
import horovod.tensorflow as hvd

# Note: this model was originally trained with conv3d layers initialized with
# TruncatedNormalInitializedVariable with stddev = 0.01.
class HorovodConvStack3DFFNModel(convstack_3d.ConvStack3DFFNModel):
  def define_tf_graph(self):
    self.show_center_slice(self.input_seed)

    if self.input_patches is None:
      self.input_patches = tf.placeholder(
          tf.float32, [1] + list(self.input_image_size[::-1]) +[1],
          name='patches')

    net = tf.concat([self.input_patches, self.input_seed], 4)

    with tf.variable_scope('seed_update', reuse=False):
      logit_update = convstack_3d._predict_object_mask(net, self.depth)

    logit_seed = self.update_seed(self.input_seed, logit_update)

    # Make predictions available, both as probabilities and logits.
    self.logits = logit_seed
    self.logistic = tf.sigmoid(logit_seed)

    if self.labels is not None:
      self.set_up_sigmoid_pixelwise_loss(logit_seed)
      self.set_up_optimizer()
      self.show_center_slice(logit_seed)
      self.show_center_slice(self.labels, sigmoid=False)
      self.add_summaries()
    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

  def set_up_optimizer_old(self, loss=None, max_gradient_entry_mag=0.7):
    """Sets up the training op for the model."""
    if loss is None:
      loss = self.loss
    tf.summary.scalar('optimizer_loss', self.loss)

    opt = optimizer.optimizer_from_flags()
    opt = hvd.DistributedOptimizer(opt)
    grads_and_vars = opt.compute_gradients(loss)

    for g, v in grads_and_vars:
      if g is None:
        tf.logging.error('Gradient is None: %s', v.op.name)

    if max_gradient_entry_mag > 0.0:
      grads_and_vars = [(tf.clip_by_value(g,
                                          -max_gradient_entry_mag,
                                          +max_gradient_entry_mag), v)
                        for g, v, in grads_and_vars]

    # TODO(b/34707785): Hopefully remove need for these deprecated calls.  Let
    # one warning through so that we have some (low) possibility of noticing if
    # the message changes.
    trainables = tf.trainable_variables()
    if trainables:
      var = trainables[0]
      tf.contrib.deprecated.histogram_summary(var.op.name, var)
    with deprecation.silence():
      for var in trainables[1:]:
        tf.contrib.deprecated.histogram_summary(var.op.name, var)
      for grad, var in grads_and_vars:
        tf.contrib.deprecated.histogram_summary(
            'gradients/' + var.op.name, grad)

    self.train_op = opt.apply_gradients(grads_and_vars,
                                        global_step=self.global_step,
                                        name='train')


  def set_up_optimizer(self, loss=None, max_gradient_entry_mag=0.7):
    """Sets up the training op for the model."""
    if loss is None:
      loss = self.loss
    tf.summary.scalar('optimizer_loss', self.loss)

    opt = optimizer.hvd_optimizer_from_flags(hvd.size())
    opt = hvd.DistributedOptimizer(opt)
    grads_and_vars = opt.compute_gradients(loss)

    for g, v in grads_and_vars:
      if g is None:
        tf.logging.error('Gradient is None: %s', v.op.name)

    if max_gradient_entry_mag > 0.0:
      grads_and_vars = [(tf.clip_by_value(g,
                                          -max_gradient_entry_mag,
                                          +max_gradient_entry_mag), v)
                        for g, v, in grads_and_vars]

    trainables = tf.trainable_variables()
    if trainables:
      for var in trainables:
        tf.summary.histogram(var.name.replace(':0', ''), var)
    for grad, var in grads_and_vars:
      tf.summary.histogram(
          'gradients/%s' % var.name.replace(':0', ''), grad)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      self.train_op = opt.apply_gradients(grads_and_vars,
                                          global_step=self.global_step,
                                          name='train')