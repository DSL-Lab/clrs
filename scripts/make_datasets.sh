venv_path=${CLRS_VENV_PATH:-/path/to/clrs_venv/bin/activate}
clrs_root=${CLRS_ROOT:-/path/to/clrs_code/clrs}
clrs_dataset_path=${CLRS_DATASET_PATH:-/path/to/save/clrs_datasets}

source ${venv_path}

# Make sure correct version of CLRS is being used
python3 -c 'import clrs; clrs.Sampler._random_er_or_k_reg_graph' || { echo 'Error: Either CLRS is not installed, or another version of CLRS is being used' ; exit 1; }

graph_algs=(2 3 7 8 9 11 20 21 27 29) # Graph Algorithm indices from specs.py, excluding articulation points and bridges
all_algs=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29) # All algorithms

### No Hint L

for alg_idx in "${all_algs[@]}"
do
for offset in 0 30 60 # Train validation test
do
    idx=$((alg_idx + offset))
    train_samples="100000"
    val_samples="32"
    test_samples="32"
    trainval_length="16"
    test_length="64"
    disable_hints="true"
    sampling_strategy="standard"
    make_dataset="True"

    echo "Generating L-CLRS for algorithm ${alg_idx} and split ${offset}"
    CLRS_MAKE_DATASET=${make_dataset} \
    CLRS_DISABLE_HINTS=${disable_hints} \
    CLRS_SAMPLING_STRATEGY=${sampling_strategy} \
    CLRS_TRAIN_SAMPLES=${train_samples} \
    CLRS_TRAIN_LENGTH=${trainval_length} \
    CLRS_VAL_SAMPLES=${val_samples} \
    CLRS_VAL_LENGTH=${trainval_length} \
    CLRS_TEST_SAMPLES=${test_samples} \
    CLRS_TEST_LENGTH=${test_length} \
        tfds build "${clrs_root}"/clrs/_src/dataset.py --data_dir "${clrs_dataset_path}"/CLRS30_${sampling_strategy}_L/ --config_idx=${idx}


    if [[ ! " ${graph_algs[*]} " =~ " ${alg_idx} " ]]; then
      continue
    fi

    train_samples="100000"
    val_samples="1000"
    test_samples="1000"
    trainval_length="16"
    test_length="32"

    sampling_strategy="reg_same_deg"

    echo "Generating L-CLRS-Len for algorithm ${alg_idx} and split ${offset}"
    CLRS_MAKE_DATASET=${make_dataset} \
    CLRS_DISABLE_HINTS=${disable_hints} \
    CLRS_SAMPLING_STRATEGY=${sampling_strategy} \
    CLRS_TRAIN_SAMPLES=${train_samples} \
    CLRS_TRAIN_LENGTH=${trainval_length} \
    CLRS_VAL_SAMPLES=${val_samples} \
    CLRS_VAL_LENGTH=${trainval_length} \
    CLRS_TEST_SAMPLES=${test_samples} \
    CLRS_TEST_LENGTH=${test_length} \
        tfds build "${clrs_root}"/clrs/_src/dataset.py --data_dir "${clrs_dataset_path}"/CLRS30_32_${sampling_strategy}_L/ --config_idx=${idx}

    sampling_strategy="reg"

    echo "Generating L-CLRS-Len-Deg for algorithm ${alg_idx} and split ${offset}"
    CLRS_MAKE_DATASET=${make_dataset} \
    CLRS_DISABLE_HINTS=${disable_hints} \
    CLRS_SAMPLING_STRATEGY=${sampling_strategy} \
    CLRS_TRAIN_SAMPLES=${train_samples} \
    CLRS_TRAIN_LENGTH=${trainval_length} \
    CLRS_VAL_SAMPLES=${val_samples} \
    CLRS_VAL_LENGTH=${trainval_length} \
    CLRS_TEST_SAMPLES=${test_samples} \
    CLRS_TEST_LENGTH=${test_length} \
        tfds build "${clrs_root}"/clrs/_src/dataset.py --data_dir "${clrs_dataset_path}"/CLRS30_32_${sampling_strategy}_L/ --config_idx=${idx}


    sampling_strategy="reg_same_nodes"
    trainval_length="32"

    echo "Generating L-CLRS-Deg for algorithm ${alg_idx} and split ${offset}"
    CLRS_MAKE_DATASET=${make_dataset} \
    CLRS_DISABLE_HINTS=${disable_hints} \
    CLRS_SAMPLING_STRATEGY=${sampling_strategy} \
    CLRS_TRAIN_SAMPLES=${train_samples} \
    CLRS_TRAIN_LENGTH=${trainval_length} \
    CLRS_VAL_SAMPLES=${val_samples} \
    CLRS_VAL_LENGTH=${trainval_length} \
    CLRS_TEST_SAMPLES=${test_samples} \
    CLRS_TEST_LENGTH=${test_length} \
        tfds build "${clrs_root}"/clrs/_src/dataset.py --data_dir "${clrs_dataset_path}"/CLRS30_32_${sampling_strategy}_L/ --config_idx=${idx}
done
done
