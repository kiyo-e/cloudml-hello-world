# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import tensorflow as tf
import multiprocessing as mp

def run_training():
  core_num = mp.cpu_count()
  config = tf.ConfigProto(
      inter_op_parallelism_threads=core_num,
      intra_op_parallelism_threads=core_num )
  sess = tf.Session(config=config)
 
  hello = tf.constant('hello, tensorflow!')
  print sess.run(hello)


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
