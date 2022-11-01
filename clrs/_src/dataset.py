# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""CLRS dataset."""
import copy
import dataclasses

import functools
import logging
import os.path
from typing import Iterator

from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

tfds.core.utils.gcs_utils._is_gcs_disabled = True


def _correct_axis_filtering(tensor, index, name):
  if 'hint_' in name:
    return tensor[:, index]
  else:
    return tensor[index]


@dataclasses.dataclass
class CLRSConfig(tfds.core.BuilderConfig):
  """Specify the split in the variant because they have different shapes."""
  split: str = ''


DEFAULT_BUILDER_CONFIGS = []


def _build_default_builder_configs():
  for split in ['train', 'val', 'test']:
    for alg in specs.CLRS_30_ALGS:
      DEFAULT_BUILDER_CONFIGS.append(
          CLRSConfig(name=f'{alg}_{split}', split=split))


_build_default_builder_configs()


class CLRSDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  BUILDER_CONFIGS = DEFAULT_BUILDER_CONFIGS

  _instantiated_dataset = None
  _instantiated_dataset_name = ''
  _instantiated_dataset_split = ''

  def _num_samples(self, algorithm_name):
    num_samples = samplers.CLRS30[self._builder_config.split]['num_samples']
    if self._builder_config.split != 'train':
      # Generate more samples for those algorithms in which the number of
      # signals is small.
      num_samples *= specs.CLRS_30_ALGS_SETTINGS[algorithm_name][
          'num_samples_multiplier']
    return num_samples

  def _create_data(self, single_sample):
    algorithm_name = '_'.join(self._builder_config.name.split('_')[:-1])
    num_samples = self._num_samples(algorithm_name)
    sampler, _ = samplers.build_sampler(
        algorithm_name,
        seed=samplers.CLRS30[self._builder_config.split]['seed'],
        num_samples=num_samples,
        length=samplers.CLRS30[self._builder_config.split]['length'],
        clrs_config=samplers.CLRS30,
        split=self._builder_config.split,
    )
    sampled_dataset = sampler.next(batch_size=1 if single_sample else None)
    data = {'input_' + t.name: t.data for t in sampled_dataset.features.inputs}
    # All other data points have input_, hint_, and output_ prefixes, so we
    # guarantee that this key is unused.
    data['lengths'] = sampled_dataset.features.lengths
    data.update({'output_' + t.name: t.data for t in sampled_dataset.outputs})
    if not sampler._clrs_config['disable_hints']:
      data.update({
          'hint_' + t.name: t.data for t in sampled_dataset.features.hints})
    self._instantiated_dataset = data

  def _info(self) -> tfds.core.DatasetInfo:
    if os.path.isfile(os.path.join(self.data_path, 'features.json')):
      return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeatureConnector.from_config(self.data_path)
      )
    else:
      assert os.environ.get('CLRS_MAKE_DATASET', False) is not False

    if (self._instantiated_dataset_name != self._builder_config.name
        or self._instantiated_dataset_split != self._builder_config.split):
      self._create_data(single_sample=True)

    data = {k: _correct_axis_filtering(v, 0, k)
            for k, v in self._instantiated_dataset.items()}
    data_info = {
        k: tfds.features.Tensor(shape=v.shape, dtype=tf.dtypes.as_dtype(
            v.dtype)) for k, v in data.items()}
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict(data_info),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    if (self._instantiated_dataset_name != self._builder_config.name
        or self._instantiated_dataset_split != self._builder_config.split):
      self._create_data(single_sample=False)
      self._instantiated_dataset_name = self._builder_config.name
      self._instantiated_dataset_split = self._builder_config.split
    return {self._builder_config.split: self._generate_examples()}

  def _generate_examples(self):
    """Generator of examples for each split."""
    algorithm_name = '_'.join(self._builder_config.name.split('_')[:-1])
    for i in range(self._num_samples(algorithm_name)):
      data = {k: _correct_axis_filtering(v, i, k)
              for k, v in self._instantiated_dataset.items()}
      yield str(i), data


def _get_clrs_file_name():
  return f'CLRS30_v{CLRSDataset.VERSION}.tar.gz'


def get_dataset_gcp_url():
  return f'https://storage.googleapis.com/dm-clrs/{_get_clrs_file_name()}'


def get_clrs_folder():
  return f'CLRS30_v{CLRSDataset.VERSION}'


def _preprocess(data_point, max_recurrent_steps, add_random_features, algorithm, seed, batch_size, exp_flags, split):
  """Convert sampled inputs into DataPoints."""
  inputs = []
  outputs = []
  hints = []
  lengths = None

  if add_random_features:
    n_nodes = data_point['input_pos'].shape[1]
    name = 'random_features'
    d_features = 16
    data = tf.random.stateless_uniform((batch_size, n_nodes, d_features), minval=-1.0, maxval=1.0, seed=seed)
    if exp_flags.orthonormal_gaussian_features: # Take orthonormal random features using QR decomposition of gaussian
      gaussian = tf.random.stateless_normal((batch_size, n_nodes, n_nodes), seed=seed)
      q, r = tf.linalg.qr(gaussian)
      data = q[:, :, :d_features]
    dp = probing.DataPoint(name, specs.Location.NODE, specs.Type.CATEGORICAL, data)
    inputs.append(dp)
  if exp_flags.add_hint_mask:
    name = 'hnt_mask'
    data = _get_hint_mask(algorithm, data_point, hint_list=exp_flags.hint_mask_list, batch_size=batch_size)
    data = tf.experimental.numpy.swapaxes(data, 0, 1)
    dp = probing.DataPoint(name, specs.Location.NODE, specs.Type.MASK, data)
    hints.append(dp)

  for name, data in data_point.items():
    if name == 'lengths':
      if max_recurrent_steps > 0: # Fixed number of steps
        data = tf.ones((batch_size, ), dtype=tf.int32) * max_recurrent_steps
        if exp_flags.stoch_depth and split == 'train':
          random_steps = tf.random.stateless_uniform(
            shape=(batch_size, ), seed=seed+1, minval=max_recurrent_steps//2, maxval=max_recurrent_steps+1, dtype=tf.int32
          )
          data = random_steps
      lengths = data
      continue
    data_point_name = name.split('_')
    name = '_'.join(data_point_name[1:])
    if name not in specs.SPECS[algorithm].keys():
      logging.warning(f"{name} not in algorithm spec, ignoring ...")
      continue
    (stage, location, dp_type) = specs.SPECS[algorithm][name]
    assert stage == data_point_name[0]
    if stage == specs.Stage.HINT:
      data = tf.experimental.numpy.swapaxes(data, 0, 1)
    dp = probing.DataPoint(name, location, dp_type, data)

    if exp_flags.edgewise_pos and name == 'pos':  # Modify to edge-wise
      assert (stage, location, dp_type) == (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR)
      n_nodes = data.shape[1]
      pairwise_pos = tf.tile(tf.experimental.numpy.expand_dims(data, axis=-1), (1, 1, n_nodes))
      data = tf.cast((pairwise_pos > tf.experimental.numpy.swapaxes(pairwise_pos, 1, 2)), dtype=tf.float32)
      dp = probing.DataPoint(name, specs.Location.EDGE, specs.Type.SCALAR, data)
    if exp_flags.random_pos and name == 'pos': # For training, randomly sample, for testing, equi-spaced
      assert (stage, location, dp_type) == (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR)
      n_nodes = data.shape[1]
      if split == 'train':
        random_pos = tf.random.stateless_uniform(
          shape=(batch_size, n_nodes), seed=seed+2, minval=0.0, maxval=1.0, dtype=data.dtype
        )
        random_pos_sorted = tf.sort(random_pos, axis=-1)
        pos_sorted_idx = tf.argsort(data, axis=-1)
        scatter_pos = tf.concat(
          [tf.tile(tf.range(0, batch_size)[:, None, None], (1, n_nodes, 1)), pos_sorted_idx[..., None]], axis=-1
        )
        data_random = tf.scatter_nd(scatter_pos, random_pos_sorted, shape=(batch_size, n_nodes))
        replace_mask = tf.tile(tf.random.stateless_binomial(
          shape=(batch_size, 1), seed=seed+3, counts=1, probs=0.5, output_dtype=tf.float64
        ), (1, n_nodes))
        data = data * replace_mask + data_random * (1 - replace_mask)
      dp = probing.DataPoint(name, location, dp_type, data)
    if exp_flags.trans_pos_enc and name == 'pos':
      n_nodes = data.shape[1]
      data = _get_transformer_positional_encoding(batch_size, n_nodes, d_model=16)
      dp = probing.DataPoint(name, location, specs.Type.CATEGORICAL, data)
    if exp_flags.pointer_hint_categorical:
      if specs.ORIGINAL_SPECS[algorithm][name] == (specs.Stage.HINT, specs.Location.NODE, specs.Type.POINTER):
        n_nodes = data_point['input_pos'].shape[1]
        data_one_hot = tf.one_hot(tf.cast(data, tf.int32), depth=n_nodes, dtype=tf.int32)
        data_cat = data_one_hot + 2 * tf.transpose(data_one_hot, (0, 1, 3, 2))
        data = tf.one_hot(data_cat, 4) # 0: no pointer, 1: one to the other, 2: other to the one, 3: both to each other
        dp = probing.DataPoint(name, specs.Location.EDGE, specs.Type.CATEGORICAL, data)

    if stage == specs.Stage.INPUT:
      inputs.append(dp)
    elif stage == specs.Stage.OUTPUT:
      outputs.append(dp)
    else:
      hints.append(dp)
  return samplers.Feedback(
      samplers.Features(tuple(inputs), tuple(hints), lengths), tuple(outputs))


def create_dataset(folder, algorithm, split, batch_size, max_recurrent_steps, add_random_features, seed, exp_flags):
  dataset = tfds.load(f'clrs_dataset/{algorithm}_{split}',
                      data_dir=folder, split=split)
  num_samples = len(dataset)  # Must be done here for correct size
  seeds = tf.data.Dataset.random(seed=seed).batch(2)
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  zip_dataset = tf.data.Dataset.zip((dataset, seeds))
  algorithm_spec = copy.deepcopy(specs.SPECS[algorithm])
  if add_random_features:
    algorithm_spec['random_features'] = (specs.Stage.INPUT, specs.Location.NODE, specs.Type.CATEGORICAL)
  if exp_flags.add_hint_mask:
    algorithm_spec['hnt_mask'] = (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK)
  if exp_flags.pointer_hint_categorical:
    for key in algorithm_spec.keys():
      if algorithm_spec[key] == (specs.Stage.HINT, specs.Location.NODE, specs.Type.POINTER):
         algorithm_spec[key] = (specs.Stage.HINT, specs.Location.EDGE, specs.Type.CATEGORICAL)
  return (
    zip_dataset.map(lambda d, s: _preprocess(
      d, algorithm=algorithm, max_recurrent_steps=max_recurrent_steps, add_random_features=add_random_features,
      seed=s, batch_size=batch_size, exp_flags=exp_flags, split=split
    )),
    num_samples,
    algorithm_spec
  )


def _copy_hint(source, dest, i, start_source, start_dest, to_add):
  """Copy from full-sample hint to a hint chunk."""
  assert np.all(dest[start_dest:, i:] == 0)
  assert start_dest < dest.shape[0]
  assert start_dest + to_add <= dest.shape[0]
  assert start_source < source.shape[0]
  assert start_source + to_add <= source.shape[0]
  dest[start_dest:start_dest+to_add, i] = source[
      start_source:start_source+to_add, i]
  return dest


def _copy_io(source, dest, i, start_dest, to_add):
  """Copy from an input or output to an input or output chunk."""
  assert np.all(dest[start_dest:, i:] == 0)
  dest[start_dest:start_dest+to_add, i] = source[i]
  return dest


def chunkify(dataset: Iterator[samplers.Feedback], chunk_length: int):
  """Generator of fixed-length chunks from full-trajectory samples.

  Args:
    dataset: full-sample dataset as numpy iterator.
    chunk_length: time length of chunks.
  Yields:
    Fixed-timelength chunks of data. Each tensor of inputs, hints and outputs
    has dimensions chunk_length x batch_size x ... Samples are not time-padded,
    after the end of one sample immediately comes the next. Since different
    samples can have different time lengths, the beginnings and ends of samples
    within a batch do not need to coincide. For this reason, the chunked
    dataset features include two chunk_length x batch_size int tensors,
    `is_first` and `is_last`, that mark the beginning and end of each sample.
    For example, if `chunk_legnth`==6 and `batch_size`==2 and the first
    full-sample batch had one sample of length 3 and one of length 5,
    we would have a first chunked batch with the following `is_first` and
    `is_last` tensors:

    is_first = [[1, 1]    is_last = [[0, 0]     ( sample id [[0 1]
                [0, 0]               [0, 0]                  [0 1]
                [0, 0]               [1, 0]                  [0 1]
                [1, 0]               [0, 0]                  [2 1]
                [0, 0]               [0, 1]                  [2 1]
                [0, 1]]              [0, 0]]                 [2 3]] )

    while the data in the inputs, outputs and hints tensors would correspond
    to samples as identified by the sample_id indicated above for reference.
    Notice that, while in the full-sample dataset inputs and outputs have
    no time dimension, here they do; the input and output tensors are simply
    repeated along each sample's time length.
  """
  def _get_batch():
    d = next(dataset)
    return (d.features.inputs, d.features.hints, d.outputs,
            d.features.lengths.astype(int))

  inputs, hints, outputs, lengths = _get_batch()
  for inp in inputs:
    if inp.location in [specs.Location.NODE, specs.Location.EDGE]:
      batch_size = inp.data.shape[0]
      break

  io_chunk = lambda x: np.zeros((chunk_length,) + x.shape, dtype=x.dtype)
  chunk_inputs = jax.tree_map(io_chunk, inputs)
  chunk_outputs = jax.tree_map(io_chunk, outputs)

  hint_chunk = lambda x: np.zeros((chunk_length,) + x.shape[1:], dtype=x.dtype)
  chunk_hints = jax.tree_map(hint_chunk, hints)

  inputs = [inputs]
  hints = [hints]
  outputs = [outputs]
  left = [lengths.copy()]
  lengths = [lengths.copy()]

  while True:
    # Create a new empty chunk
    chunk_inputs = jax.tree_map(np.zeros_like, chunk_inputs)
    chunk_hints = jax.tree_map(np.zeros_like, chunk_hints)
    chunk_outputs = jax.tree_map(np.zeros_like, chunk_outputs)
    start_mark = np.zeros((chunk_length, batch_size), dtype=int)
    end_mark = np.zeros((chunk_length, batch_size), dtype=int)

    # Get enough data batches to fill the new chunk
    while np.any(np.sum(left, axis=0) < chunk_length):
      inp, hh, out, ll = _get_batch()
      inputs.append(inp)
      hints.append(hh)
      outputs.append(out)
      left.append(ll.copy())
      lengths.append(ll.copy())

    # Fill the chunk, one batch element at a time
    for i in range(batch_size):
      total, idx = 0, 0
      while total < chunk_length:
        to_add = min(left[idx][i], chunk_length - total)
        if to_add:
          start = lengths[idx][i] - left[idx][i]
          assert start >= 0
          f_io = functools.partial(_copy_io, i=i, start_dest=total,
                                   to_add=to_add)
          chunk_inputs = jax.tree_map(f_io, inputs[idx], chunk_inputs)
          chunk_outputs = jax.tree_map(f_io, outputs[idx], chunk_outputs)
          f_hint = functools.partial(_copy_hint, i=i, start_source=start,
                                     start_dest=total, to_add=to_add)
          chunk_hints = jax.tree_map(f_hint, hints[idx], chunk_hints)
          if start == 0:
            start_mark[total, i] = 1
          total += to_add
          left[idx][i] -= to_add
          assert left[idx][i] >= 0
          if left[idx][i] == 0:
            end_mark[total - 1, i] = 1
        idx += 1
      assert total == chunk_length

    while left and np.all(left[0] == 0):
      inputs.pop(0)
      hints.pop(0)
      outputs.pop(0)
      left.pop(0)
      lengths.pop(0)

    yield samplers.Feedback(
        samplers.FeaturesChunked(chunk_inputs, chunk_hints,
                                 start_mark, end_mark),
        chunk_outputs)


def create_chunked_dataset(folder, algorithm, split, batch_size, chunk_length):
  dataset = tfds.load(f'clrs_dataset/{algorithm}_{split}',
                      data_dir=folder, split=split)
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(lambda d: _preprocess(d, algorithm=algorithm))
  dataset = dataset.as_numpy_iterator()
  return chunkify(dataset, chunk_length), specs.SPECS[algorithm]


def _get_hint_mask(algorithm, data_point, hint_list, batch_size):
  if hint_list == 'all':
    hint_list = None
  else:
    hint_list = hint_list.split('/')

  n_nodes = data_point['input_pos'].shape[1]
  mask = None
  for name, data in data_point.items():
    if name == 'lengths':
      continue
    data_point_name = name.split('_')
    name = '_'.join(data_point_name[1:])
    stage = data_point_name[0]
    if stage != specs.Stage.HINT:
      continue
    if (hint_list is None) or (name in hint_list):
      if mask is None:
        mask = tf.Variable(lambda : tf.zeros((batch_size, data.shape[1], n_nodes), dtype=tf.bool), name='mask')
        mask[:, 0, :].assign(tf.ones((batch_size, n_nodes), dtype=tf.bool))

      data_change = (tf.math.abs((data[:, 1:, ...] - data[:, :-1, ...])) > 0.00001)
      (stage, location, dp_type) = specs.ORIGINAL_SPECS[algorithm][name]
      if location == specs.Location.NODE:
        if dp_type in [specs.Type.MASK, specs.Type.MASK_ONE, specs.Type.SCALAR]:
          pass
        elif dp_type == specs.Type.POINTER:
          data_one_hot = tf.one_hot(tf.cast(data, tf.int32), depth=n_nodes)
          data_change = (tf.math.abs((data_one_hot[:, 1:, ...] - data_one_hot[:, :-1, ...])) > 0.00001)
          data_change = tf.math.logical_or(
            tf.math.reduce_any(data_change, -1),
            tf.math.reduce_any(tf.transpose(data_change, (0, 1, 3, 2)), -1)
          )
        elif dp_type == specs.Type.CATEGORICAL:
          data_change = tf.reduce_any(data_change, -1)
        else:
          raise Exception("Nah!")
      elif location == specs.Location.EDGE:
        if dp_type in [specs.Type.MASK, specs.Type.MASK_ONE, specs.Type.SCALAR]:
          data_change = tf.math.logical_or(
            tf.math.reduce_any(data_change, -1),
            tf.math.reduce_any(tf.transpose(data_change, (0, 1, 3, 2)), -1)
          )
        else:
          raise Exception("Nah!!")
      elif location == specs.Location.GRAPH:
        raise Exception("No graph hints please!")
      assert tuple(data_change.shape[1:]) == (mask.shape[1] - 1, n_nodes, ), (data_change.shape, mask.shape, name)
      mask[:, 1:, ...].assign(tf.math.logical_or(mask[:, 1:, ...], data_change))

  assert mask is not None
  # return tf.ones((batch_size, mask.shape[1], n_nodes), dtype=tf.bool)
  mask = _time_convolve_mask(mask)
  return mask

def _time_convolve_mask(mask):
  conv_mask = tf.Variable(lambda: tf.zeros_like(mask), name='mask_conv')
  conv_mask.assign(mask)
  conv_mask[:, 1:, ...].assign(tf.math.logical_or(mask[:, :-1, ...], conv_mask[:, 1:, ...]))
  conv_mask[:, :-1, ...].assign(tf.math.logical_or(mask[:, 1:, ...], conv_mask[:, :-1, ...]))
  return conv_mask

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
  return pos * angle_rates

def _get_transformer_positional_encoding(batch_size, position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  pos_encoding = np.tile(angle_rads[np.newaxis, ...], (batch_size, 1, 1))
  return tf.cast(pos_encoding, dtype=tf.float32)
