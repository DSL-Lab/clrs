import hashlib
import logging

import jax
try: # Only needed for calling functions in this file
  import networkx as nx
  import matplotlib.pyplot as plt
except:
  pass
import numpy as np
import jax.numpy as jnp

import tensorflow_datasets as tfds
import pickle
import clrs

def compare_dataset_io(file_name1, file_name2, algorithm, split):
  dataset1 = tfds.load(f'clrs_dataset/{algorithm}_{split}',
                       data_dir=file_name1, split=split).as_numpy_iterator()
  dataset2 = tfds.load(f'clrs_dataset/{algorithm}_{split}',
                       data_dir=file_name2, split=split).as_numpy_iterator()
  ok = 0
  for d1, d2 in zip(dataset1, dataset2):
    for k in d1.keys():
      assert (d1[k]-d2[k]).sum() < 0.0000001, f"Mis Match!, {k}, {d1[k]}, {d2[k]}, {ok}, {algorithm}"
      ok += 1
  print(f"OK: {ok}")


def load_and_save_logits(rng_key, eval_model: clrs.models.BaselineModel, param_file: str, sampler, sample_count, spec,
                         batch_size, file_name, save_file=True):
  eval_model.restore_model(param_file, replace_params=True)
  predict_fn = eval_model.predict

  processed_samples = 0
  inputs = []
  preds = []
  hint_preds = []
  outputs = []
  output_logits = []
  hints = []
  lengths = []
  while processed_samples < sample_count:
    feedback = next(sampler)
    inputs.append(feedback.features.inputs)
    outputs.append(feedback.outputs)
    rng_key, new_rng_key = jax.random.split(rng_key)
    cur_preds, (cur_hint_preds, _, _, hiddens), net_aux = predict_fn(rng_key, feedback.features)
    output_logits.append(net_aux['output_logits'])
    preds.append(cur_preds)
    lengths.append(feedback.features.lengths)
    hints.append(feedback.features.hints)
    hint_preds.append(cur_hint_preds)
    rng_key = new_rng_key
    processed_samples += batch_size
  outputs = _concat(outputs, axis=0)
  output_logits = _concat(output_logits, axis=0)
  preds = _concat(preds, axis=0)
  lengths = _concat(lengths, axis=0)
  inputs = _concat(inputs, axis=0)
  # for hints, axis=1 because hints have time dimension first
  hints = _concat(hints, axis=1)
  # for hint_preds, axis=0 because the time dim is unrolled as a list
  hint_preds = _concat(hint_preds, axis=0)
  if save_file:
    jnp.save(f'/data/smahdavi/tmp/{file_name}.npy', dict(
      inputs=inputs, preds=preds, outputs=outputs, hints=hints, lengths=lengths, hint_preds=hint_preds,
      output_logits=output_logits
    ))
  return preds, outputs, hints, lengths, hint_preds, spec

def _concat(dps, axis):
  return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis), *dps)


def load_and_save_model_soup(param_files: list, output_file:str, coefs: list):
  param_states = []
  opt_state = []
  for file_name in param_files:
    with open(file_name, 'rb') as f:
      restored_state = pickle.load(f)
      param_states.append(restored_state['params'])
      opt_state = restored_state['opt_state']
  param_state = jax.tree_util.tree_map(lambda *x: sum([x_ * coef_ for (x_, coef_) in zip(x, coefs)]), *param_states)
  to_save = {'params': param_state, 'opt_state': opt_state}
  path = output_file
  with open(path, 'wb') as f:
    pickle.dump(to_save, f)



def visualize_graph(A: jnp.ndarray, bridges: jnp.ndarray, pred_bridges: jnp.ndarray):
  A = A - jnp.diag(A.diagonal())
  G = nx.from_numpy_array(A).to_undirected()
  plt.figure(figsize=(20, 20))
  pos = nx.spring_layout(G, seed=1, scale=2, k=0.25)

  if bridges is None:
    bridges = A * 2 - 1
    logging.warning("Replacing bridges with all edges")

  pred_bridges = jnp.where(bridges != -1, pred_bridges, bridges)
  bridge_list = [(node1.item(), node2.item()) for (node1, node2) in zip(*jnp.where(bridges == 1))]
  pred_bridges_list = [(node1.item(), node2.item()) for (node1, node2) in zip(*jnp.where(pred_bridges == 1))]
  print(bridge_list)
  print(G.edges())
  nx.draw_networkx_nodes(G, pos)
  nx.draw_networkx_labels(G, pos)
  nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', alpha=0.5)
  nx.draw_networkx_edges(G, pos, edgelist=bridge_list, edge_color='green', style='dashed')
  nx.draw_networkx_edges(G, pos, edgelist=pred_bridges_list, edge_color='red', style='dashed')
  plt.savefig('/tmp/1.png')
  plt.close()


def _generate_bridge_graphs(seed, n_nodes, p=0.3): # A Bit larger P
  np_rng = np.random.RandomState(seed)

  def _random_connected_graph(rng):
    while True:
      mat = rng.binomial(1, p, size=(n_nodes//2, n_nodes//2))
      mat *= np.transpose(mat)
      G = nx.from_numpy_array(mat).to_undirected()
      if nx.is_connected(G):
        break
    return mat

  mat1 = _random_connected_graph(np_rng)
  mat2 = _random_connected_graph(np_rng)
  block_with_bridge = np.zeros((n_nodes//2, n_nodes//2))
  block_with_bridge[np_rng.random_integers(0, n_nodes//2-1), np_rng.random_integers(0, n_nodes//2-1)] = 1
  block_no_bridge = np.copy(block_with_bridge)
  while True:
    block_no_bridge[np_rng.random_integers(0, n_nodes//2-1), np_rng.random_integers(0, n_nodes//2-1)] = 1
    if ((block_no_bridge - block_with_bridge) ** 2).sum() > 0.5:
      break

  mat_bridge = np.block([
    [mat1, block_with_bridge],
    [block_with_bridge.T, mat1],
  ])
  mat_no_bridge = np.block([
    [mat2, block_no_bridge],
    [block_no_bridge.T, mat2],
  ])
  output = np.block([
    [np.zeros((n_nodes//2, n_nodes//2)), block_with_bridge],
    [block_with_bridge.T, np.zeros((n_nodes//2, n_nodes//2))],
  ]) * 2 - 1 # 0 -> -1

  return mat_bridge, mat_no_bridge, output

def _add_self_loops(mat):
  adj = np.copy(mat)
  np.fill_diagonal(adj, 1)
  return adj

def _get_feedback_sampler(example_feedbacks, n_iters):
  batch_size, n_nodes = example_feedbacks.features.inputs[0].data.shape[:2]
  for i in range(n_iters):
    A_batch = np.zeros_like(example_feedbacks.features.inputs[1].data)
    adj_batch = np.zeros_like(A_batch)
    out_batch = np.zeros((batch_size // 2, n_nodes, n_nodes))
    for j in range(batch_size//2):
      mat_bridge, mat_no_bridge, output = _generate_bridge_graphs(i, n_nodes)
      adj_bridge, adj_no_bridge = _add_self_loops(mat_bridge), _add_self_loops(mat_no_bridge)
      A_batch[2*j] = mat_bridge
      A_batch[2*j+1] = mat_no_bridge
      adj_batch[2*j] = adj_bridge
      adj_batch[2*j+1] = adj_no_bridge
      out_batch[j] = output
    inputs = []
    for inp in example_feedbacks.features.inputs:
      if inp.name == 'A':
        data = A_batch
      elif inp.name == 'adj':
        data = adj_batch
      else:
        data = inp.data
      inputs.append(clrs.DataPoint(inp.name, inp.location, inp.type_, data))
    feedback = clrs.Feedback(
      clrs.Features(
        tuple(inputs), tuple(), example_feedbacks.features.lengths
      ), np.stack(out_batch)
    )
    yield feedback
  return

def _eval_bridge_outputs(outputs, preds):
  b, n, _ = outputs.shape
  b2 = preds['is_bridge'].data.shape[0]
  assert 2 * b == b2
  correct = 0
  for i in range(b):
    x_bridge = jnp.where(outputs[i] == 1, preds['is_bridge'].data[2 * i], 0).sum()
    x_no_bridge = jnp.where(outputs[i] == 1, preds['is_bridge'].data[2 * i + 1], 0).sum()
    print(
      i,
      x_bridge,
      x_no_bridge,
    )
    if (x_bridge > 0.5) and (x_no_bridge < 0.5):
      correct += 1
  print(correct / b)

def eval_bridges(rng_key, eval_model: clrs.models.BaselineModel, param_file: str, sampler, sample_count, spec,
                         batch_size, file_name, save_file=True):
  eval_model.restore_model(param_file, replace_params=True)
  predict_fn = eval_model.predict

  processed_samples = 0
  inputs = []
  preds = []
  hint_preds = []
  outputs = []
  output_logits = []
  hints = []
  lengths = []
  example_feedback = next(sampler)
  for feedback in _get_feedback_sampler(example_feedback, n_iters=125):
    inputs.append(feedback.features.inputs)
    outputs.append(feedback.outputs)
    rng_key, new_rng_key = jax.random.split(rng_key)
    cur_preds, (cur_hint_preds, _, _, hiddens), net_aux = predict_fn(rng_key, feedback.features)
    output_logits.append(net_aux['output_logits'])
    preds.append(cur_preds)
    lengths.append(feedback.features.lengths)
    hints.append(feedback.features.hints)
    hint_preds.append(cur_hint_preds)
    rng_key = new_rng_key
  outputs = _concat(outputs, axis=0)
  output_logits = _concat(output_logits, axis=0)
  preds = _concat(preds, axis=0)
  lengths = _concat(lengths, axis=0)
  inputs = _concat(inputs, axis=0)
  _eval_bridge_outputs(outputs, preds)

  # print("Hi")
  # vis_idx = 71*2
  # visualize_graph(inputs[1].data[vis_idx], outputs[vis_idx // 2], preds['is_bridge'].data[vis_idx])
  # visualize_graph(inputs[1])
  return
