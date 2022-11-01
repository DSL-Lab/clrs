# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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

"""Run a full test run for one or more algorithmic tasks from CLRS."""

import os


os.environ["WANDB_MODE"] = "dryrun"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

import time
from absl import app
from absl import flags
from absl import logging

import clrs
import jax
import jax.numpy as jnp
import numpy as np
import string
import tensorflow as tf
from ml_collections import config_flags, ConfigDict
import wandb

flags.DEFINE_string('algorithm', 'bfs', 'Which algorithm to run.')
flags.DEFINE_integer('seed', 42, 'Random seed to set')

flags.DEFINE_integer('batch_size', 32, 'Batch size used for training.')
flags.DEFINE_boolean('chunked_training', False,
                     'Whether to use chunking for training.')
flags.DEFINE_integer('chunk_length', 100,
                     'Time chunk length used for training (if '
                     '`chunked_training` is True.')
flags.DEFINE_integer('train_items', 160000,
                     'Number of items (i.e., individual examples, possibly '
                     'repeated) processed during training. With non-chunked'
                     'training, this is the number of training batches times '
                     'the number of training steps. For chunked training, '
                     'as many chunks will be processed as needed to get these '
                     'many full examples.')
flags.DEFINE_integer('eval_every', 320,
                     'Logging frequency (in training examples).')
flags.DEFINE_boolean('verbose_logging', False, 'Whether to log aux losses.')

flags.DEFINE_integer('hidden_size', 128,
                     'Number of hidden size units of the model.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate to use.')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')
flags.DEFINE_float('hint_teacher_forcing_noise', 0.0,
                   'Probability that rematerialized hints are encoded during '
                   'training instead of ground-truth teacher hints. Only '
                   'pertinent in encoded_decoded modes.')
flags.DEFINE_integer('nb_heads', 1, 'Number of heads for GAT processors')

flags.DEFINE_enum('hint_mode', 'none',
                  ['encoded_decoded', 'decoded_only',
                   'encoded_decoded_nodiff', 'decoded_only_nodiff',
                   'none'])

flags.DEFINE_boolean('use_ln', True,
                     'Whether to use layer normalisation in the processor.')
flags.DEFINE_boolean('use_lstm', False,
                     'Whether to insert an LSTM after message passing.')
flags.DEFINE_enum(
    'processor_type', 'mpnn',
    ['deepsets', 'mpnn', 'pgn', 'pgn_mask',
     'gat', 'gatv2', 'gat_full', 'gatv2_full',
     'memnet_full', 'memnet_masked', 'mynet', 'edge_att', 'hybrid', 'pgn_mpnn', 'gat_edge'],
    'The processor type to use.')

flags.DEFINE_string('checkpoint_path', '/tmp/CLRS30',
                    'Path in which checkpoints are saved.')
flags.DEFINE_string('dataset_path', '/tmp/CLRS30',
                    'Path in which dataset is stored.')
flags.DEFINE_string('log_path', '/tmp/',
                    'Path in which logs are stored.')
flags.DEFINE_boolean('freeze_processor', False,
                     'Whether to freeze the processor of the model.')

flags.DEFINE_string('log_prefix', 'test', 'Log Prefix')
flags.DEFINE_enum('scheduler', 'cosine', ['none', 'lr_plateau', 'cosine'], 'Scheduler')
flags.DEFINE_integer(
  'max_recurrent_steps', 32, 'Fix the number of recurrent steps to max_recurrent_steps regardless of the underlying '
                            'algorithm number of steps. 0 for variable (taken from the input datapoints)'
)
flags.DEFINE_boolean('add_random_features', True,
                     'Whether to add random features to each datapoint.')
flags.DEFINE_boolean('debug', False, 'Activate debugging.')
flags.DEFINE_enum('discretizer', 'none', ['none', 'binary', 'vq', 'gumbel'], 'What discretizer to use?')
flags.DEFINE_boolean('clip_grad', False, 'Clip Gradient Norm?')
flags.DEFINE_string('wandb_project', 'clrs_project', 'WandB project name.')
flags.DEFINE_string('wandb_team', 'clrs_team', 'WandB team name.')
config_flags.DEFINE_config_dict('exp_flags', ConfigDict({
  'apply_mask': True, # Apply mask to 2WL processor
  'orthonormal_gaussian_features': True,
  'random_pos': False, # Random Position Scalar
  'trans_pos_enc': False, # Transformer Positional Encoding
  'edgewise_pos': False, # Edgewise Pos. Enc.
  'infrequent_test_eval': False, # Some tasks contain a large test dataset, this flag evaluates test dataset of such tasks less frequently.
  'hybrid_processors': 'p_e', # PGN_MPNN & Edge Attention
  'add_ln_transformer': False,
  'hint_list': 'all', # Only keep some hints
  'add_hint_mask': False, #
  'hint_mask_list': 'all', # List of hints to keep track of their change, only relevant if add_hint_mask activated
  'hybrid_type': 'sigmoid', # Hybrid type, choose between "avg", and "sigmoid"
  'token_route': True,
  'pointer_hint_categorical': False,
  'stoch_depth': False, # Instead of fixed number of recurrent steps, use a stochastic depth
  'markovian_processing': False, # Do not carry over a hidden, rely on hints from the previous step only
  'mpnn_reduction': 'max',
}))

FLAGS = flags.FLAGS

RUN_ID = ''.join(np.random.choice(np.array(list(string.ascii_lowercase)), size=10))


def unpack(v):
  try:
    return v.item()  # DeviceArray
  except (AttributeError, ValueError):
    return v


def evaluate(rng_key, model, feedback, spec, extras=None, verbose=False):
  """Evaluates a model on feedback."""
  out = {}
  predictions, aux, net_aux = model.predict(rng_key, feedback.features)
  out.update(clrs.evaluate(feedback.outputs, predictions))
  if model.decode_hints and verbose:
    hint_preds = [clrs.decoders.postprocess(spec, x) for x in aux[0]]
    out.update(clrs.evaluate_hints(feedback.features.hints,
                                   feedback.features.lengths,
                                   hint_preds))
  if extras:
    out.update(extras)
  if verbose:
    out.update(model.verbose_loss(feedback, aux))
  return {k: unpack(v) for k, v in out.items()}


def evaluate_preds(preds, outputs, hints, lengths, hint_preds, spec, extras):
  """Evaluates predictions against feedback."""
  out = {}
  out.update(clrs.evaluate(outputs, preds))
  if hint_preds:
    hint_preds = [clrs.decoders.postprocess(spec, x) for x in hint_preds]
    out.update(clrs.evaluate_hints(hints, lengths, hint_preds))
  if extras:
    out.update(extras)
  return {k: unpack(v) for k, v in out.items()}


def _concat(dps, axis):
  return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis), *dps)


def collect_and_eval(sampler, predict_fn, sample_count, rng_key, spec, extras):
  """Collect batches of output and hint preds and evaluate them."""
  verbose = FLAGS.verbose_logging
  processed_samples = 0
  inputs = []
  preds = []
  hint_preds = []
  outputs = []
  hints = []
  lengths = []
  while processed_samples < sample_count:
    feedback = next(sampler)
    inputs.append(feedback.features.inputs)
    outputs.append(feedback.outputs)
    rng_key, new_rng_key = jax.random.split(rng_key)
    cur_preds, (cur_hint_preds, _, _, hiddens), net_aux = predict_fn(rng_key, feedback.features)
    # print((net_aux['pgn_route'] > 0.).mean(), (net_aux['token_gt_route'] > 0.).mean())
    preds.append(cur_preds)
    lengths.append(feedback.features.lengths)
    if verbose:
      hints.append(feedback.features.hints)
      hint_preds.append(cur_hint_preds)
    rng_key = new_rng_key
    processed_samples += FLAGS.batch_size
  outputs = _concat(outputs, axis=0)
  preds = _concat(preds, axis=0)
  lengths = _concat(lengths, axis=0)
  inputs = _concat(inputs, axis=0)

  if verbose:
    # for hints, axis=1 because hints have time dimension first
    hints = _concat(hints, axis=1)
    # for hint_preds, axis=0 because the time dim is unrolled as a list
    hint_preds = _concat(hint_preds, axis=0)

  return evaluate_preds(preds, outputs, hints, lengths, hint_preds, spec,
                        extras)


def get_dataset_folder():
  """Downloads CLRS30 dataset if not already downloaded."""
  dataset_folder = FLAGS.dataset_path
  if os.path.isdir(dataset_folder):
    logging.info('Dataset found at %s.', dataset_folder)
    return dataset_folder
  raise Exception("No Local Dataset! Please download and extract it manually and then try again.")


def _config_logging(is_debug):
  log_path = os.path.join(FLAGS.log_path, FLAGS.log_prefix)
  os.makedirs(log_path, exist_ok=True)
  if not is_debug:
      logging.get_absl_handler().use_absl_log_file(
          f'{FLAGS.algorithm}_{FLAGS.seed}',
          log_path
      )


def _config_wandb(log_prefix, absl_flags):
  debug = absl_flags.debug
  wandb_run = wandb.init(
    project=absl_flags.wandb_project, entity=absl_flags.wandb_team, mode="disabled" if debug else None,
    config={
      'prefix': FLAGS.log_prefix,
      'exp_id': FLAGS.log_prefix.split('/')[0],
      'run_id': RUN_ID,
      'dataset_id': absl_flags.dataset_path.split('/')[-1]
    }
  )
  wandb.config.update(absl_flags)
  wandb.run.name = f"{log_prefix}/{RUN_ID}"
  wandb.run.save()
  return wandb_run


def main(unused_argv):
  exp_flags = FLAGS.exp_flags
  if exp_flags.hint_list != "all":
    _prune_hints(exp_flags.hint_list, FLAGS.algorithm)
  if FLAGS.hint_mode == 'none':  # Remove all hints
    _prune_hints("", FLAGS.algorithm)
  _config_logging(is_debug=FLAGS.debug)
  # Use canonical CLRS-30 samplers.
  clrs30_spec = clrs.CLRS30
  logging.info(f'Run id: {RUN_ID}')
  logging.info('Using CLRS30 spec: %s', clrs30_spec)
  logging.info('Using flags: %s', FLAGS.flag_values_dict())
  wandb_run = _config_wandb(FLAGS.log_prefix, FLAGS)
  dataset_folder = get_dataset_folder()
  check_flag_combinations(FLAGS)
  training_steps = FLAGS.train_items // FLAGS.batch_size

  if FLAGS.hint_mode == 'encoded_decoded_nodiff':
    encode_hints = True
    decode_hints = True
    decode_diffs = False
  elif FLAGS.hint_mode == 'decoded_only_nodiff':
    encode_hints = False
    decode_hints = True
    decode_diffs = False
  elif FLAGS.hint_mode == 'encoded_decoded':
    encode_hints = True
    decode_hints = True
    decode_diffs = True
  elif FLAGS.hint_mode == 'decoded_only':
    encode_hints = False
    decode_hints = True
    decode_diffs = True
  elif FLAGS.hint_mode == 'none':
    encode_hints = False
    decode_hints = False
    decode_diffs = False
  else:
    raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')

  common_args = dict(folder=dataset_folder,
                     algorithm=FLAGS.algorithm,
                     max_recurrent_steps=FLAGS.max_recurrent_steps,
                     add_random_features=FLAGS.add_random_features,
                     batch_size=FLAGS.batch_size,
                     seed=FLAGS.seed,
                     exp_flags=exp_flags
                     )
  # Make full dataset pipeline run on CPU (including prefetching).
  with tf.device('/cpu:0'):
    if FLAGS.chunked_training:
      train_sampler, spec = clrs.create_chunked_dataset(
          **common_args, split='train', chunk_length=FLAGS.chunk_length)
      train_sampler_for_eval, _, _ = clrs.create_dataset(
          split='train', **common_args)
      train_sampler_for_eval = train_sampler_for_eval.as_numpy_iterator()
    else:
      train_sampler, _, spec = clrs.create_dataset(**common_args, split='train')
      train_sampler = train_sampler.as_numpy_iterator()
      train_sampler_for_eval = None

    val_sampler, val_samples, _ = clrs.create_dataset(
        **common_args, split='val')
    val_sampler = val_sampler.as_numpy_iterator()
    test_sampler, test_samples, _ = clrs.create_dataset(
        **common_args, split='test')
    test_sampler = test_sampler.as_numpy_iterator()

  processor_factory = clrs.get_processor_factory(
    FLAGS.processor_type,
    use_ln=FLAGS.use_ln,
    nb_heads=FLAGS.nb_heads,
    has_graph='adj' in spec.keys(),
    exp_flags=exp_flags,
  )
  net_params = dict(
    max_recurrent_steps=FLAGS.max_recurrent_steps,
    algorithm=FLAGS.algorithm,
    discretizer=FLAGS.discretizer,
  )
  model_params = dict(
      processor_factory=processor_factory,
      hidden_dim=FLAGS.hidden_size,
      encode_hints=encode_hints,
      decode_hints=decode_hints,
      decode_diffs=decode_diffs,
      use_lstm=FLAGS.use_lstm,
      learning_rate=FLAGS.learning_rate,
      scheduler=FLAGS.scheduler,
      training_steps=training_steps,
      checkpoint_path=FLAGS.checkpoint_path,
      freeze_processor=FLAGS.freeze_processor,
      dropout_prob=FLAGS.dropout_prob,
      hint_teacher_forcing_noise=FLAGS.hint_teacher_forcing_noise,
      net_params=net_params,
      clip_grad=FLAGS.clip_grad,
      exp_flags=exp_flags,
  )

  eval_model = clrs.models.BaselineModel(
      spec=spec,
      dummy_trajectory=next(val_sampler),
      **model_params
  )
  if FLAGS.chunked_training:
    train_model = clrs.models.BaselineModelChunked(
        spec=spec,
        dummy_trajectory=next(train_sampler),
        **model_params
        )
  else:
    train_model = eval_model

  # Training loop.
  best_score = -1.0  # Ensure that there is overwriting
  rng_key = jax.random.PRNGKey(FLAGS.seed)
  current_train_items = 0
  step = 0
  next_eval = 0

  val_score = test_score = -1
  while current_train_items < FLAGS.train_items:
    feedback = next(train_sampler)

    # Initialize model.
    if current_train_items == 0:
      t = time.time()
      train_model.init(feedback.features, FLAGS.seed + 1)

    # Training step step.
    rng_key, new_rng_key = jax.random.split(rng_key)
    cur_loss = train_model.feedback(rng_key, feedback)
    rng_key = new_rng_key
    if current_train_items == 0:
      logging.info('Compiled feedback step in %f s.', time.time() - t)
      # logging.info(f'Number of parameters: {sum(x.size for x in jax.tree_leaves(train_model.params))}')
    if FLAGS.chunked_training:
      examples_in_chunk = jnp.sum(feedback.features.is_last)
    else:
      examples_in_chunk = len(feedback.features.lengths)
    current_train_items += examples_in_chunk

    # Periodically evaluate model.
    if current_train_items >= next_eval:
      common_extras = {'examples_seen': current_train_items,
                       'step': step}
      eval_model.params = train_model.params
      # Training info.
      if FLAGS.chunked_training:
        train_feedback = next(train_sampler_for_eval)
      else:
        train_feedback = feedback
      rng_key, new_rng_key = jax.random.split(rng_key)
      train_stats = evaluate(
          rng_key,
          eval_model,
          train_feedback,
          spec=spec,
          extras=dict(loss=cur_loss, **common_extras),
          verbose=FLAGS.verbose_logging,
      )
      rng_key = new_rng_key
      train_score = train_stats['score']
      train_loss = train_stats['loss']
      logging.info('(train) step %d: %s', step, train_stats)

      # Validation info.
      rng_key, new_rng_key = jax.random.split(rng_key)
      val_stats = collect_and_eval(
          val_sampler,
          eval_model.predict,
          val_samples,
          rng_key,
          spec=spec,
          extras=common_extras)
      rng_key = new_rng_key
      val_score = val_stats['score']
      logging.info('(val) step %d: %s', step, val_stats)

      # Here we evaluate test, but do not use it anywhere, just for the "accuracy on the line" reference.
      if exp_flags.infrequent_test_eval and (FLAGS.algorithm in ['quickselect', 'minimum', 'binary_search', 'naive_string_matcher',
            'kmp_matcher', 'segments_intersect', 'find_maximum_subarray_kadane']) and (step % 5000 != 0):
        test_score = -1.0
      else:
        rng_key, new_rng_key = jax.random.split(rng_key)
        test_stats = collect_and_eval(
          test_sampler,
          eval_model.predict,
          test_samples,
          rng_key,
          spec=spec,
          extras=common_extras)
        rng_key = new_rng_key
        test_score = test_stats['score']
        logging.info('(test) step %d: %s', step, test_stats)

      # If best scores, update checkpoint.
      score = val_stats['score']
      if score + 0.000001 > best_score:  # Validation accuracies equal -> Keep the last one
        logging.info('Saving new checkpoint...')
        best_score = score
        train_model.save_model(f'best_{RUN_ID}.pkl')
      next_eval += FLAGS.eval_every
      wandb.log({
        'train_score': train_score,
        'train_score_glevel': train_stats['graph_score'],
        'val_score': val_score,
        'val_score_glevel': val_stats['graph_score'],
        'test_score': test_score,
        'test_score_glevel': test_stats['graph_score'],
        'train_loss': train_loss,
      }, step=step)
    step += 1

  # Training complete, evaluate on test set.
  logging.info('Restoring best model from checkpoint...')
  eval_model.restore_model(f'best_{RUN_ID}.pkl', only_load_processor=False)

  rng_key, new_rng_key = jax.random.split(rng_key)
  test_stats = collect_and_eval(
      test_sampler,
      eval_model.predict,
      test_samples,
      rng_key,
      spec=spec,
      extras=common_extras)
  rng_key = new_rng_key
  best_val_test_score = test_stats['score']
  logging.info('(test) step %d: %s', step, test_stats)

  wandb.run.summary.update({
    'best_val': best_score,
    'best_test': best_val_test_score,
    'best_test_glevel': test_stats['graph_score'],
    'last_val': val_score,
    'last_test': test_score,
  })
  wandb_run.finish()

def _prune_hints(remaining_hints, algorithm):
  hint_list = remaining_hints.split('/')
  for key, (stage, _, _) in list(clrs.SPECS[algorithm].items()):
    if (key not in hint_list) and (stage == clrs.Stage.HINT):
      clrs.SPECS[algorithm].pop(key)


def check_flag_combinations(absl_flags):
  assert not absl_flags.chunked_training, "No Support for chunked training"
  if absl_flags.discretizer == 'vq':
    assert absl_flags.processor_type not in ['edge_att', 'hybrid'], "Quantizer is not implemented for edge processors"
  assert sum(
    [absl_flags.exp_flags.edgewise_pos, absl_flags.exp_flags.random_pos, absl_flags.exp_flags.trans_pos_enc]
  ) <= 1, "Only one positional change can be made"


if __name__ == '__main__':
  app.run(main)
