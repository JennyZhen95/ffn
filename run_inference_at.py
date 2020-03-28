#!/usr/bin/env python

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
"""Runs FFN inference within a dense bounding box.

Inference is performed within a single process.
"""

import os
import time

from google.protobuf import text_format
from absl import app
from absl import flags
gfile = tf.io.gfile

from ffn.utils import bounding_box_pb2
from ffn.inference import inference
from ffn.inference import inference_flags
from ffn.inference import storage
from ffn.inference import seed

from scipy.special import expit
import itertools
import numpy as np
FLAGS = flags.FLAGS

flags.DEFINE_string('bounding_box', None,
                    'BoundingBox proto in text format defining the area '
                    'to segmented.')
flags.DEFINE_list('start_pos', None, 'start pos in z,y,x')
# class dummyPolicy(seed.BaseSeedPolicy):
#   def __init__(self, canvas, start_pos, **kwargs):
#     super(dummyPolicy).__init__(self, canvas, **kwargs)
#     self.start_pos = s
#   def __next__(self):
#     for i in [self.start_pos]:
#       return i
  
def main(unused_argv):
  request = inference_flags.request_from_flags()

  if not gfile.exists(request.segmentation_output_dir):
    gfile.makedirs(request.segmentation_output_dir)

  bbox = bounding_box_pb2.BoundingBox()
  text_format.Parse(FLAGS.bounding_box, bbox)

  # start_pos = tuple([int(i) for i in FLAGS.start_pos])
  runner = inference.Runner()

  corner = (bbox.start.z, bbox.start.y, bbox.start.x)
  subvol_size = (bbox.size.z, bbox.size.y, bbox.size.x)
  start_pos = tuple([int(i) for i in FLAGS.start_pos])

  seg_path = storage.segmentation_path(
      request.segmentation_output_dir, corner)
  prob_path = storage.object_prob_path(
      request.segmentation_output_dir, corner)

  runner.start(request)
  canvas, alignment = runner.make_canvas(corner, subvol_size)
  num_iter = canvas.segment_at(start_pos)

  print('>>', num_iter)

  sel = [slice(max(s, 0), e + 1) for s, e in zip(
      canvas._min_pos - canvas._pred_size // 2,
      canvas._max_pos + canvas._pred_size // 2)]
  mask = canvas.seed[sel] >= canvas.options.segment_threshold
  raw_segmented_voxels = np.sum(mask)

  mask &= canvas.segmentation[sel] <= 0
  actual_segmented_voxels = np.sum(mask)
  canvas._max_id += 1
  canvas.segmentation[sel][mask] = canvas._max_id
  canvas.seg_prob[sel][mask] = storage.quantize_probability(
      expit(canvas.seed[sel][mask]))

  runner.save_segmentation(canvas, alignment, seg_path, prob_path)

  runner.run((bbox.start.z, bbox.start.y, bbox.start.x),
             (bbox.size.z, bbox.size.y, bbox.size.x))

  counter_path = os.path.join(request.segmentation_output_dir, 'counters.txt')
  if not gfile.exists(counter_path):
    runner.counters.dump(counter_path)


if __name__ == '__main__':
  app.run(main)
