# coding=utf-8
# This code is modified based on generativ_evale.py at 
#
#     https://github.com/google-research/google-research/tree/master/genomics_ood
#
# Copyright 2021 University of Southern California
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

r"""Evaluating OOD detection for generative model-based methods.

    LSTM likelihood for MLR-OOD.

    Bai, Xin, et al. "MLR-OOD: a Markov chain based Likelihood Ratio method for Out-Of-Distribution detection of genomic sequences."

    Ren, Jie, et al. "Likelihood Ratios for Out-of-Distribution Detection."
    arXiv preprint arXiv:1906.02845 (2019).


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import numpy as np
import tensorflow as tf

import train
from genomics_ood import utils

TPR_THRES = 0.8

flags.DEFINE_string(
    'model_dir_frgd',
    '/tmp/out_training',
    'Directory to ckpts of the generative models.')
flags.DEFINE_integer('n_samples', '10000', 'Number of sequences for evaluation')
flags.DEFINE_integer('ckpt_step', '900000', 'The step of the selected ckpt.')
flags.DEFINE_string(
    'test_fasta_data',
    '/tmp/data/test/test.fa',
    'fasta file of testing sequences')
flags.DEFINE_string('test_data_dir', '/tmp/data/test',
                    'data directory of testing tfrecord data')
flags.DEFINE_integer('testing_seq_len', 250, 'testing sequence length.')
flags.DEFINE_integer('training_seq_len', 250, 'training sequence length.')
flags.DEFINE_string('out_test_likelihood_file', '/tmp/out_eval/lstm_test_likelihood_class0.txt',
                    'Directory where to write log and models.')

FLAGS = flags.FLAGS

def list_to_np(list_batch):
  # list_batch is a list of np arrays, each np array is of length batch_size
  return np.stack(list_batch).reshape(-1)

def restore_model_from_ckpt(ckpt_dir, ckpt_file):
  """restore model from ckpt file."""
  # load params
  params_json_file = os.path.join(ckpt_dir, 'params.json')
  params = utils.generate_hparams(params_json_file)
  params.in_val_data_dir = FLAGS.test_data_dir

  # create model
  tf.reset_default_graph()
  model = train.SeqModel(params)
  model.reset()
  model.restore_from_ckpt(ckpt_file)
  return params, model

def load_test_dataset(test_data_dir, test_file_pattern):
  """load the test dataset."""
  test_data_file_list = [
    os.path.join(test_data_dir, x)
    for x in tf.gfile.ListDirectory(test_data_dir)
    if test_file_pattern in x and '.tfrecord' in x
  ]
  tf.logging.info('test_data_file_list=%s', test_data_file_list)
  test_dataset = tf.data.TFRecordDataset(
    test_data_file_list).map(lambda v: utils.parse_single_tfexample(v, v))
  return test_dataset

dict = {"A":'0', "C":'1', "G":'2', "T":'3'}
def get_example_object(data_record):
  # Convert individual data into a list of int64 or float or bytes
  int_list1 = tf.train.Int64List(value = [data_record['int_data']])
  str_list1 = tf.train.BytesList(value = [data_record['str_data'].encode('utf-8')])
  # Create a dictionary with above lists individually wrapped in Feature
  feature_key_value_pair = {
    'x': tf.train.Feature(bytes_list = str_list1),
    'y': tf.train.Feature(int64_list = int_list1)
  }
  features = tf.train.Features(feature = feature_key_value_pair)
  example = tf.train.Example(features = features)
  return example

def testing_fasta_to_tfrecord(fasta_file, test_data_dir, training_seq_len, testing_seq_len):
  if testing_seq_len % training_seq_len != 0:
    raise ValueError('The testing sequence length must be a multiple of the training sequence length!')
  if not os.path.exists(fasta_file):
    e = 'File %s do no exsits!'%fasta_file
    raise Exception(e)
  if not test_data_dir.endswith('/'):
    test_data_dir += '/'
    test_tfrecord_file_name = test_data_dir+ 'test_len' + str(testing_seq_len) + '.tfrecord'
  with tf.python_io.TFRecordWriter(test_tfrecord_file_name) as tfwriter:
    with open(fasta_file) as f:
      for line in f.readlines():
        if not line.startswith('>'):
          line = line.strip()
          if len(line) != testing_seq_len:
            raise ValueError('Each sequence must have the same length as the inputtesting sequence length!')
          numeric_line = [dict[x] if x in dict else x for x in line]
          str_numeric_line = ' '.join(numeric_line)
          for idx in range(testing_seq_len // training_seq_len):
            seq = str_numeric_line[idx * training_seq_len : (idx + 1) * training_seq_len]
            data_record = {
              'str_data': seq,
              # we use -1 for the label of all testing sequences, in reality, the user may have knowledge on the testing sequence labels, but we do not expect our users to have that knowledge.
              'int_data': -1
            }
            example = get_example_object(data_record)
            tfwriter.write(example.SerializeToString())

def write_test_likelihood(test_ll, out_test_likelihood_file, n_samples):
  np.savetxt(out_test_likelihood_file, test_ll.reshape((n_samples, )))

def main(_):
  model_dir = {'frgd': FLAGS.model_dir_frgd}
  # placeholders
  ll_test = {}

  # generate the testing tfrecord data according to the testing fasta file
  fasta_file = FLAGS.test_fasta_data
  test_data_dir = FLAGS.test_data_dir
  training_seq_len = FLAGS.training_seq_len
  testing_seq_len = FLAGS.testing_seq_len
  testing_fasta_to_tfrecord(fasta_file, test_data_dir, training_seq_len, testing_seq_len)

  # evaluation on test
  for key in ['frgd']:

    _, ckpt_file = utils.get_ckpt_at_step(model_dir[key], FLAGS.ckpt_step)
    if not ckpt_file:
      tf.logging.fatal('%s model ckpt not exist', ckpt_file)
    params, model = restore_model_from_ckpt(model_dir[key], ckpt_file)

    # specify test datasets for test
    params.test_file_pattern = 'test'

    test_dataset = load_test_dataset(FLAGS.test_data_dir, params.test_file_pattern)
    print(tf.gfile.ListDirectory(FLAGS.test_data_dir))

    loss_test, _, _, y_test, _ = model.pred_from_ckpt(
        test_dataset, FLAGS.n_samples)

    ll_test[key] = -list_to_np(loss_test)  # full model likelihood ratio

  write_test_likelihood(ll_test['frgd'], FLAGS.out_test_likelihood_file, FLAGS.n_samples)


if __name__ == '__main__':
  tf.app.run()
