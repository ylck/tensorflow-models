# Copyright 2016 Google Inc. All Rights Reserved.
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
# pylint: disable=line-too-long
"""A binary to train Inception in a distributed manner using multiple systems.

Please see accompanying README.md for details and instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception import inception_distributed_train
from inception.imagenet_data import ImagenetData
import json
import os

FLAGS = tf.app.flags.FLAGS


def main(unused_args):
  # Extract all the hostnames for the ps and worker jobs to construct the
  # cluster spec.

  tf_config_json = os.environ.get("TF_CONFIG", "{}")
  tf_config = json.loads(tf_config_json)
  tf.logging.info("tf_config: %s", tf_config)

  task = tf_config.get("task", {})
  tf.logging.info("task: %s", task)

  cluster_spec = tf_config.get("cluster", {})
  tf.logging.info("cluster_spec: %s", cluster_spec)

  if cluster_spec:
      cluster_spec_object = tf.train.ClusterSpec(cluster_spec)
      server_def = tf.train.ServerDef(
          cluster=cluster_spec_object.as_cluster_def(),
          protocol="grpc",
          job_name=task["type"],
          task_index=task["index"])

      tf.logging.info("server_def: %s", server_def)

      tf.logging.info("Building server.")
      # Create and start a server for the local task.
      server = tf.train.Server(server_def)
      tf.logging.info("Finished building server.")

  if task["type"] == 'ps':
    # `ps` jobs wait for incoming connections from the workers.
    server.join()
  else:
    # `worker` jobs will actually do the work.
    dataset = ImagenetData(subset=FLAGS.subset)
    assert dataset.data_files()
    # Only the chief checks for or creates train_dir.
    if task["index"] == 0:
      if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    inception_distributed_train.train(server.target, dataset, cluster_spec, task["index"])

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
