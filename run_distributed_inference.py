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
import timeit

from google.protobuf import text_format
from absl import app
from absl import flags
from tensorflow import gfile
import os
import numpy as np

from ffn.utils import bounding_box_pb2
from ffn.inference import inference
from ffn.inference import inference_flags
from ffn.utils import bounding_box
from ffn.utils import geom_utils
from tensorflow.python.client import device_lib

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

FLAGS = flags.FLAGS

flags.DEFINE_string('bounding_box', None,
                    'BoundingBox proto in text format defining the area '
                    'to segmented.')
flags.DEFINE_list('subvolume_size', '512,512,128', '"valid"subvolume_size to issue to each runner')
flags.DEFINE_list('overlap', '32,32,16', 'overlap of bbox')
flags.DEFINE_boolean('use_cpu', True, 'Use CPU instead of GPU')
flags.DEFINE_boolean('distribute_gpu', True, 'Allocate on different GPUs')

def divide_bounding_box(bbox, subvolume_size, overlap):
  """divide up into valid subvolumes."""
  # deal with parsed bbox missing "end" attr
  start = geom_utils.ToNumpy3Vector(bbox.start)
  size = geom_utils.ToNumpy3Vector(bbox.size)

  bbox = bounding_box.BoundingBox(start, size)
  print(bbox)

  calc = bounding_box.OrderlyOverlappingCalculator(
    outer_box=bbox, 
    sub_box_size=subvolume_size, 
    overlap=overlap, 
    include_small_sub_boxes=True,
    back_shift_small_sub_boxes=False)
  
  return [bb for bb in calc.generate_sub_boxes()]

def main(unused_argv):
  start_time = time.time()
  # mpi version

  request = inference_flags.request_from_flags()
  if mpi_rank == 0:
    if not gfile.Exists(request.segmentation_output_dir):
      gfile.MakeDirs(request.segmentation_output_dir)

    bbox = bounding_box_pb2.BoundingBox()
    text_format.Parse(FLAGS.bounding_box, bbox)
    print(request)

    subvolume_size = np.array([int(i) for i in FLAGS.subvolume_size])
    overlap = np.array([int(i) for i in FLAGS.overlap])
    sub_bboxes = divide_bounding_box(bbox, subvolume_size, overlap)
    sub_bboxes = np.array_split(np.array(sub_bboxes), mpi_size)
    print(sub_bboxes)
  else:
    sub_bboxes = None
  
  sub_bboxes = mpi_comm.scatter(sub_bboxes, 0)
  print('rank %d, bbox: %s' % (mpi_rank, len(sub_bboxes)))
  
  for sub_bbox in sub_bboxes:
    out_name = 'seg-%d_%d_%d_%d_%d_%d' % (
      sub_bbox.start[0], sub_bbox.start[1], sub_bbox.start[2], 
      sub_bbox.size[0], sub_bbox.size[1], sub_bbox.size[2])
    segmentation_output_dir = os.path.join(request.segmentation_output_dir, out_name)
    request.segmentation_output_dir = segmentation_output_dir
    use_gpu = str(mpi_rank % 2)
    runner = inference.Runner(use_cpu=FLAGS.use_cpu, use_gpu=use_gpu)
    runner.start(request)
    runner.run(sub_bbox.start[::-1], sub_bbox.size[::-1])

  end_time = time.time()
  print('>> ', end_time - start_time)
if __name__ == '__main__':
  app.run(main)
