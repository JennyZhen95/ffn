import re
from pprint import pprint
import argparse
import os
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.training import training
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.ops import variables

def repl(match_obj):
  if match_obj.group(0) == 'biases': return 'bias'



def migrate_checkpoint(old_ckpt, new_ckpt):
  os.makedirs(os.path.dirname(new_ckpt), exist_ok=True)
  var_map = {}
  var_names_map = {}
  with ops.Graph().as_default():
    reader = training.NewCheckpointReader(old_ckpt)
    variable_names = sorted(reader.get_variable_to_shape_map())
    
    for name_v1 in variable_names:
      if 'biases' in name_v1:
        name_v2 = re.sub('biases', 'bias', name_v1)
      elif 'weights' in name_v1:
        name_v2 = re.sub('weights', 'kernel', name_v1)
      else:
        name_v2 = name_v1
      print('%s -> %s' % (name_v1, name_v2))
      var_names_map[name_v1] = name_v2
    for name_v1 in variable_names:
      tensor_v1 = reader.get_tensor(name_v1)
      name_v2 = var_names_map[name_v1]
      if name_v1 == 'global_step':
        tensor_v1 = tf.cast(tensor_v1, dtype=tf.int64)
      tensor_v2 = tf.Variable(tensor_v1, name=name_v2)
      var_map[name_v2] = tensor_v2
    pprint(var_map)
    saver = saver_lib.Saver(var_list=var_map)
    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
      saver.save(sess, new_ckpt)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--old_checkpoint', type=str, help='tf1 checkpoint')
  parser.add_argument('--new_checkpoint', type=str, help='tf2 checkpoint')
  args = parser.parse_args()
  # old_ckpt = '/home/hanyu/workspace/ffn_migrate/checkpoint_migrate/old_checkpoints/model.ckpt-353787448'
  # new_ckpt = '/home/hanyu/workspace/ffn_migrate/checkpoint_migrate/new_checkpoints/model.ckpt-353787448'
  migrate_checkpoint(args.old_checkpoint, args.new_checkpoint)
if __name__ == '__main__':
  main()