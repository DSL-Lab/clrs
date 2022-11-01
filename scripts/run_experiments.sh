TMPDIR=${TMPDIR:-/tmp}
data_root=${CLRS_DATASET_PATH:-/path/to/saved/clrs_datasets}
log_root=${CLRS_LOG_PATH:-/tmp/clrs_logs}
checkpoint_root=${CLRS_CHECKPOINT_PATH:-/tmp/clrs_checkpoints}
steps=${steps:-20000}

all_algorithms=("articulation_points" "activity_selector" "bellman_ford" "bfs" "binary_search" "bridges" "bubble_sort" "dag_shortest_paths" "dfs" "dijkstra" "find_maximum_subarray_kadane" "floyd_warshall" "graham_scan" "heapsort" "insertion_sort" "jarvis_march" "kmp_matcher" "lcs_length" "matrix_chain_order" "minimum" "mst_kruskal" "mst_prim" "naive_string_matcher" "optimal_bst" "quickselect" "quicksort" "segments_intersect" "strongly_connected_components" "task_scheduling" "topological_sort")
distinct_algorithms=("articulation_points" "activity_selector" "bellman_ford" "bfs" "binary_search" "bridges" "dag_shortest_paths" "dfs" "find_maximum_subarray_kadane" "floyd_warshall" "graham_scan" "lcs_length" "matrix_chain_order" "minimum" "mst_kruskal" "mst_prim" "naive_string_matcher" "optimal_bst" "quickselect" "quicksort" "segments_intersect" "strongly_connected_components" "task_scheduling" "topological_sort")

# Processor Comparisons, Table 5 Row 1
# mpnn processor is MPNN-FC in the paper
# pgn_mpnn processor is MPNN-G in the paper
# edge_att processor is 2WL in the paper
for algorithm in "${distinct_algorithms[@]}"
do
  batch_size_mp=32
  batch_size_2wl=16
  dataset="${data_root}/CLRS30_standard_L"
  train_items_mp=$(( steps*batch_size_mp ))
  train_items_2wl=$(( steps*batch_size_2wl ))
  exp_name="main_table2"
  hidden_size=128
  hidden_size_hybrid=108 # Stay in the same parameter budget
  hint_mode="none"
  for seed in {42..44}
  do
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/pgn_mpnn --seed=${seed} --batch_size=${batch_size_mp} --processor_type pgn_mpnn --hint_mode=${hint_mode} --hidden_size ${hidden_size} --algorithm ${algorithm} --train_items ${train_items_mp} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/2wl_2 --seed=${seed} --batch_size=${batch_size_2wl} --exp_flags.infrequent_test_eval=True --processor_type edge_att --hint_mode=${hint_mode} --hidden_size ${hidden_size} --algorithm ${algorithm} --train_items ${train_items_2wl} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/hybrid_avg_2 --seed=${seed} --exp_flags.hybrid_type=avg --exp_flags.infrequent_test_eval=True --batch_size=${batch_size_2wl} --processor_type hybrid --hint_mode=${hint_mode} --hidden_size ${hidden_size_hybrid} --algorithm ${algorithm} --train_items ${train_items_2wl} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/hybrid_sigmoid --seed=${seed} --exp_flags.hybrid_type=sigmoid --exp_flags.infrequent_test_eval=True --batch_size=${batch_size_2wl} --processor_type hybrid --hint_mode=${hint_mode} --hidden_size ${hidden_size_hybrid} --algorithm ${algorithm} --train_items ${train_items_2wl} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/hybrid_avg_pp --seed=${seed} --exp_flags.hybrid_processors=p_p --exp_flags.hybrid_type=avg --exp_flags.infrequent_test_eval=True --batch_size=${batch_size_2wl} --processor_type hybrid --hint_mode=${hint_mode} --hidden_size ${hidden_size_hybrid} --algorithm ${algorithm} --train_items ${train_items_2wl} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/hybrid_avg_ee --seed=${seed} --exp_flags.hybrid_processors=e_e --exp_flags.hybrid_type=avg --exp_flags.infrequent_test_eval=True --batch_size=${batch_size_2wl} --processor_type hybrid --hint_mode=${hint_mode} --hidden_size ${hidden_size_hybrid} --algorithm ${algorithm} --train_items ${train_items_2wl} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
  done
done

# Processors + Random Scalar Position, Table 5 row 2
for algorithm in "${distinct_algorithms[@]}"
do
  batch_size_mp=32
  batch_size_2wl=16
  dataset="${data_root}/CLRS30_standard_L"
  train_items_mp=$(( steps*batch_size_mp ))
  train_items_2wl=$(( steps*batch_size_2wl ))
  exp_name="final_results"
  hidden_size=128
  hidden_size_hybrid=108 # Stay in the same parameter budget
  hint_mode="none"
  for seed in {42..44}
  do
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/pgn_mpnn --exp_flags.random_pos=True --seed=${seed} --batch_size=${batch_size_mp} --processor_type pgn_mpnn --hint_mode=${hint_mode} --hidden_size ${hidden_size} --algorithm ${algorithm} --train_items ${train_items_mp} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/2wl_2 --exp_flags.random_pos=True --seed=${seed} --batch_size=${batch_size_2wl} --exp_flags.infrequent_test_eval=True --processor_type edge_att --hint_mode=${hint_mode} --hidden_size ${hidden_size} --algorithm ${algorithm} --train_items ${train_items_2wl} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/hybrid_avg_2 --exp_flags.random_pos=True --seed=${seed} --exp_flags.hybrid_type=avg --exp_flags.infrequent_test_eval=True --batch_size=${batch_size_2wl} --processor_type hybrid --hint_mode=${hint_mode} --hidden_size ${hidden_size_hybrid} --algorithm ${algorithm} --train_items ${train_items_2wl} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/hybrid_sigmoid --exp_flags.random_pos=True --seed=${seed} --exp_flags.hybrid_type=sigmoid --exp_flags.infrequent_test_eval=True --batch_size=${batch_size_2wl} --processor_type hybrid --hint_mode=${hint_mode} --hidden_size ${hidden_size_hybrid} --algorithm ${algorithm} --train_items ${train_items_2wl} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/hybrid_avg_pp --exp_flags.random_pos=True --seed=${seed} --exp_flags.hybrid_processors=p_p --exp_flags.hybrid_type=avg --exp_flags.infrequent_test_eval=True --batch_size=${batch_size} --processor_type hybrid --hint_mode=${hint_mode} --hidden_size ${hidden_size_hybrid} --algorithm ${algorithm} --train_items ${train_items_2wl} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/hybrid_avg_ee --exp_flags.random_pos=True --seed=${seed} --exp_flags.hybrid_processors=e_e --exp_flags.hybrid_type=avg --exp_flags.infrequent_test_eval=True --batch_size=${batch_size} --processor_type hybrid --hint_mode=${hint_mode} --hidden_size ${hidden_size_hybrid} --algorithm ${algorithm} --train_items ${train_items_2wl} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
  done
done


# Position Encoding Ablations, Tables 4 and 6
for algorithm in "naive_string_matcher" "dfs" "find_maximum_subarray_kadane" "dag_shortest_paths"  "matrix_chain_order" "topological_sort" "bfs"
do
  batch_size=32
  dataset="${data_root}/CLRS30_standard_L"
  train_items=$(( steps*batch_size ))
  exp_name="pos_encodings_ablation2"
  hidden_size=128
  hint_mode="none"
  for seed in {42..44}
  do
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/pgn_mpnn/standard --seed=${seed} --batch_size=${batch_size} --processor_type pgn_mpnn --hint_mode=${hint_mode} --hidden_size ${hidden_size} --algorithm ${algorithm} --train_items ${train_items} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/pgn_mpnn/random_pos --seed=${seed} --exp_flags.random_pos=True  --batch_size=${batch_size} --processor_type pgn_mpnn --hint_mode=${hint_mode} --hidden_size ${hidden_size} --algorithm ${algorithm} --train_items ${train_items} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/pgn_mpnn/trans_pos_enc --seed=${seed} --exp_flags.trans_pos_enc=True  --batch_size=${batch_size} --processor_type pgn_mpnn --hint_mode=${hint_mode} --hidden_size ${hidden_size} --algorithm ${algorithm} --train_items ${train_items} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/mpnn/standard --seed=${seed} --batch_size=${batch_size} --processor_type mpnn --hint_mode=${hint_mode} --hidden_size ${hidden_size} --algorithm ${algorithm} --train_items ${train_items} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/mpnn/edgewise_pos --seed=${seed} --exp_flags.edgewise_pos=True --batch_size=${batch_size} --processor_type mpnn --hint_mode=${hint_mode} --hidden_size ${hidden_size} --algorithm ${algorithm} --train_items ${train_items} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/mpnn/random_pos --seed=${seed} --exp_flags.random_pos=True --batch_size=${batch_size} --processor_type mpnn --hint_mode=${hint_mode} --hidden_size ${hidden_size} --algorithm ${algorithm} --train_items ${train_items} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
  done
done

# Generalization Modes Ablations, Table 2
# CLRS30_32_reg_same_nodes_L is when number of nodes stay the same and corresponds to L-CLRS-Deg in the paper
# CLRS30_32_reg_same_deg_L is when degree stays the same, and corresponds to L-CLRS-Len in the paper
# CLRS30_32_reg_L is when both degree and number of nodes change, and corresponds to L-CLRS-Len-Deg in the paper
for algorithm in "bellman_ford" "bfs" "dag_shortest_paths" "dfs" "floyd_warshall" "mst_kruskal" "mst_prim" "strongly_connected_components" "topological_sort"
do
  batch_size=8
  train_items=$(( steps*batch_size ))
  exp_name="ood_mode_table2"
  hint_mode="none"
  hidden_size=256
  for seed in {42..44}
  do
    for dataset_name in "CLRS30_32_reg_same_nodes_L" "CLRS30_32_reg_L" "CLRS30_32_reg_same_deg_L"
    do
      dataset="${data_root}/${dataset_name}"
      python3 -m clrs.examples.run --log_prefix ${exp_name}/${algorithm}/pgn_mpnn_${dataset_name} --batch_size=${batch_size} --processor_type pgn_mpnn --hint_mode=${hint_mode} --hidden_size ${hidden_size} --seed=${seed} --algorithm ${algorithm} --train_items ${train_items} --checkpoint_path "${checkpoint_root}" --dataset_path "${dataset}" --log_path "${log_root}"
    done
  done
done
