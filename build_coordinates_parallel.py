#!/usr/bin/env python

"""Builds a TFRecord file of coordinates for training.

Use ./compute_partitions.py to generate data for --partition_volumes.
Note that the volume names you provide in --partition_volumes will
have to match the volume labels you pass to the training script.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from absl import app
from absl import flags
from absl import logging

import h5py
import numpy as np
import tensorflow as tf

import sys
from tqdm import tqdm

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.rank
mpi_size = mpi_comm.size


FLAGS = flags.FLAGS

flags.DEFINE_list('partition_volumes', None,
                  'Partition volumes as '
                  '<volume_name>:<volume_path>:<dataset>, where volume_path '
                  'points to a HDF5 volume, and <volume_name> is an arbitrary '
                  'label that will have to also be used during training.')
flags.DEFINE_string('coordinate_output', None,
                    'Path to a TF Record file in which to save the '
                    'coordinates.')
flags.DEFINE_list('margin', None, '(z, y, x) tuple specifying the '
                  'number of voxels adjacent to the border of the volume to '
                  'exclude from sampling. This should normally be set to the '
                  'radius of the FFN training FoV (i.e. network FoV radius '
                  '+ deltas.')
flags.DEFINE_integer('max_samples', 100000, 'Max number of samples for each '
                  'partition')


IGNORE_PARTITION = 255


def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(argv):
  del argv  # Unused.

  if mpi_rank == 0:
    totals = defaultdict(int)  # partition -> voxel count
    indices = defaultdict(list)  # partition -> [(vol_id, 1d index)]

    vol_labels = []
    vol_shapes = []
    mz, my, mx = [int(x) for x in FLAGS.margin]

    for i, partvol in enumerate(FLAGS.partition_volumes):
      name, path, dataset = partvol.split(':')
      with h5py.File(path, 'r') as f:
        partitions = f[dataset][mz:-mz, my:-my, mx:-mx]
        vol_shapes.append(partitions.shape)
        vol_labels.append(name)

        uniques, counts = np.unique(partitions, return_counts=True)
        for val, cnt in zip(uniques, counts):
          if val == IGNORE_PARTITION:
            continue

          totals[val] += cnt
          indices[val].extend(
              [(i, flat_index) for flat_index in
              np.flatnonzero(partitions == val)])

    logging.info('Partition counts:')
    for k, v in totals.items():
      logging.info(' %d: %d', k, v)

    # check time
    # logging.info('indices %s', str(indices[3][0:100]))
    # test_output = np.resize(indices[3][0:100], (1000,2))
    # logging.info('resampled %s', str(test_output))

    max_count = max(totals.values())
    # keys = list(totals.keys())
    # keys.sort()

    # logging.info(keys)
    # indices_list = [ indices[k] for k in keys ]
    # print(len(indices_list))

    #indices = np.concatenate(
    #    [np.resize(np.random.permutation(v), (max_count, 2)) for
    #    v in tqdm(indices.values())], axis=0)
    #indices_list = [[np.resize(np.random.permutation(v), (max_count, 2)) for
    #    v in tqdm(indices.values())]

    indices_list = []
    #max_count = FLAGS.max_samples
    for v in tqdm(indices.values()):
      if len(v) < FLAGS.max_samples:
        indices_list.append( np.resize(np.random.permutation(v), (FLAGS.max_samples, 2)))
      else:
        indices_list.append( np.stack([v[i] for i in np.random.choice(len(v), FLAGS.max_samples, False)], 0))
      #logging.info(indices_list[-1].shape)
        
    indices = np.concatenate(indices_list, 0)
    np.random.shuffle(indices)
    logging.info('length %s', indices.shape)
    logging.info('Finished.')
    subset_indices = np.array_split(indices, mpi_size)
    #logging.info('size %s', subset_indices[0].shape)
    #indices_subset = indices[0]
  else:
    subset_indices =  None
    vol_shapes = None
    vol_labels = None
    mx = None
    my = None
    mz = None


  subset_indices = mpi_comm.scatter(subset_indices, 0)
  #indices_list = mpi_comm.scatter(indices_list, 0)
  vol_shapes = mpi_comm.bcast(vol_shapes, 0)
  vol_labels = mpi_comm.bcast(vol_labels, 0)
  mx = mpi_comm.bcast(mx, 0)
  my = mpi_comm.bcast(my, 0)
  mz = mpi_comm.bcast(mz, 0)

  logging.info('Shape %d %s.', mpi_rank, subset_indices.shape)
  mpi_comm.barrier()

  sharded_fname = '%s-%s-of-%s' % (FLAGS.coordinate_output, str(mpi_rank).zfill(5), str(mpi_size).zfill(5))
  logging.info(sharded_fname)

  record_options = tf.python_io.TFRecordOptions(
      tf.python_io.TFRecordCompressionType.GZIP)
  with tf.python_io.TFRecordWriter(sharded_fname,
                                   options=record_options) as writer:
    for i, coord_idx in tqdm(subset_indices):
      z, y, x = np.unravel_index(coord_idx, vol_shapes[i])

      coord = tf.train.Example(features=tf.train.Features(feature=dict(
          center=_int64_feature([mx + x, my + y, mz + z]),
          label_volume_name=_bytes_feature(vol_labels[i].encode('utf-8'))
      )))
      writer.write(coord.SerializeToString())


if __name__ == '__main__':
  flags.mark_flag_as_required('margin')
  flags.mark_flag_as_required('coordinate_output')
  flags.mark_flag_as_required('partition_volumes')

  app.run(main)
