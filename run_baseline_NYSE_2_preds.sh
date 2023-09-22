# Copyright 2020 The Google Research Authors.
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

# Modifications copyright (C) 2021 <NUS/Cui Can>

while getopts a:p:m:s:v:t:e:i:c:n:d:g:l: flag
do
    case "${flag}" in
        a) init=${OPTARG};;
        p) the_output_path_for_evolved_alphas_and_performance=${OPTARG};;
        m) my_alpha_file_path=${OPTARG};;
        s) steps=${OPTARG};;
        v) cutoff_valid=${OPTARG};;
        t) cutoff_test=${OPTARG};;
        e) test_efficiency=${OPTARG};;
        i) predict_index=${OPTARG};;
        c) predict_index_confidence=${OPTARG};;
        n) num_top_stocks=${OPTARG};;
        d) try_random_seeds=${OPTARG};;
        g) generate_preds_data=${OPTARG};;
        l) all_timesteps=${OPTARG};;
    esac
done

DATA_DIR=$(pwd)/data_for_EvoAlgo_kdd2023_predsNYSE # 

bazel --output_user_root=/hdd9/james/folder_to_make_space_for_home/bazel/_bazel_james8 run -c opt \
  --copt=-DMAX_SCALAR_ADDRESSES=30 \
  --copt=-DMAX_VECTOR_ADDRESSES=30 \
  --copt=-DMAX_MATRIX_ADDRESSES=25 \
    :run_search_experiment -- \
      --the_output_path_for_evolved_alphas_and_performance=$the_output_path_for_evolved_alphas_and_performance \
      --my_alpha_file_path=$my_alpha_file_path \
      --cutoff_valid="$cutoff_valid" \
      --cutoff_test="$cutoff_test" \
      --test_efficiency=${test_efficiency:-0} \
      --predict_index=${predict_index:-0} \
      --predict_index_confidence=${predict_index_confidence:-0.6} \
      --num_top_stocks=${num_top_stocks:-50} \
      --generate_preds_data="$generate_preds_data" \
      --all_timesteps=${all_timesteps:-1228} \
      --search_experiment_spec=" \
        search_tasks { \
          tasks { \
            stock_task { \
              dataset_name: 'NYSE' \
              path: '${DATA_DIR}' \
              max_supported_data_seed: 1 \
            } \
            features_size: 13 \
            num_train_examples: 1000 \
            num_valid_examples: 228 \
            num_train_epochs: 1 \
            num_tasks: 1402 \
            eval_type: RMS_ERROR\
          } \
        } \
        setup_ops: [SCALAR_CONST_SET_OP, VECTOR_CONST_SET_OP, MATRIX_CONST_SET_OP, SCALAR_UNIFORM_SET_OP, VECTOR_UNIFORM_SET_OP, MATRIX_UNIFORM_SET_OP, SCALAR_GAUSSIAN_SET_OP, VECTOR_GAUSSIAN_SET_OP, MATRIX_GAUSSIAN_SET_OP] \
        predict_ops: [NO_OP, SCALAR_SUM_OP, SCALAR_DIFF_OP, SCALAR_PRODUCT_OP, SCALAR_DIVISION_OP, SCALAR_ABS_OP, SCALAR_RECIPROCAL_OP, SCALAR_SIN_OP, SCALAR_COS_OP, SCALAR_TAN_OP, SCALAR_ARCSIN_OP, SCALAR_ARCCOS_OP, SCALAR_ARCTAN_OP, SCALAR_EXP_OP, SCALAR_LOG_OP, SCALAR_HEAVYSIDE_OP, VECTOR_HEAVYSIDE_OP, MATRIX_HEAVYSIDE_OP, SCALAR_VECTOR_PRODUCT_OP, SCALAR_BROADCAST_OP, VECTOR_RECIPROCAL_OP, VECTOR_NORM_OP, VECTOR_ABS_OP, VECTOR_SUM_OP, VECTOR_DIFF_OP, VECTOR_PRODUCT_OP, VECTOR_DIVISION_OP, VECTOR_INNER_PRODUCT_OP, VECTOR_OUTER_PRODUCT_OP, SCALAR_MATRIX_PRODUCT_OP, MATRIX_RECIPROCAL_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_COLUMN_BROADCAST_OP, VECTOR_ROW_BROADCAST_OP, MATRIX_NORM_OP, MATRIX_COLUMN_NORM_OP, MATRIX_ROW_NORM_OP, MATRIX_TRANSPOSE_OP, MATRIX_ABS_OP, MATRIX_SUM_OP, MATRIX_DIFF_OP, MATRIX_PRODUCT_OP, MATRIX_DIVISION_OP, MATRIX_MATRIX_PRODUCT_OP, SCALAR_MIN_OP, VECTOR_MIN_OP, MATRIX_MIN_OP, SCALAR_MAX_OP, VECTOR_MAX_OP, MATRIX_MAX_OP, VECTOR_MEAN_OP, MATRIX_MEAN_OP, MATRIX_ROW_MEAN_OP, MATRIX_ROW_ST_DEV_OP, VECTOR_ST_DEV_OP, MATRIX_ST_DEV_OP, VECTOR_UNIFORM_SET_OP, MATRIX_UNIFORM_SET_OP, CORRELATION_OP, MATRIX_GET_ROW_OP, MATRIX_GET_COLUMN_OP, COVARIANCE_OP, MATRIX_GET_SCALAR_OP] \
        learn_ops: [NO_OP, SCALAR_SUM_OP, SCALAR_DIFF_OP, SCALAR_PRODUCT_OP, SCALAR_DIVISION_OP, SCALAR_ABS_OP, SCALAR_RECIPROCAL_OP, SCALAR_SIN_OP, SCALAR_COS_OP, SCALAR_TAN_OP, SCALAR_ARCSIN_OP, SCALAR_ARCCOS_OP, SCALAR_ARCTAN_OP, SCALAR_EXP_OP, SCALAR_LOG_OP, SCALAR_HEAVYSIDE_OP, VECTOR_HEAVYSIDE_OP, MATRIX_HEAVYSIDE_OP, SCALAR_VECTOR_PRODUCT_OP, SCALAR_BROADCAST_OP, VECTOR_RECIPROCAL_OP, VECTOR_NORM_OP, VECTOR_ABS_OP, VECTOR_SUM_OP, VECTOR_DIFF_OP, VECTOR_PRODUCT_OP, VECTOR_DIVISION_OP, VECTOR_INNER_PRODUCT_OP, VECTOR_OUTER_PRODUCT_OP, SCALAR_MATRIX_PRODUCT_OP, MATRIX_RECIPROCAL_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_COLUMN_BROADCAST_OP, VECTOR_ROW_BROADCAST_OP, MATRIX_NORM_OP, MATRIX_COLUMN_NORM_OP, MATRIX_ROW_NORM_OP, MATRIX_TRANSPOSE_OP, MATRIX_ABS_OP, MATRIX_SUM_OP, MATRIX_DIFF_OP, MATRIX_PRODUCT_OP, MATRIX_DIVISION_OP, MATRIX_MATRIX_PRODUCT_OP, SCALAR_MIN_OP, VECTOR_MIN_OP, MATRIX_MIN_OP, SCALAR_MAX_OP, VECTOR_MAX_OP, MATRIX_MAX_OP, VECTOR_MEAN_OP, MATRIX_MEAN_OP, MATRIX_ROW_MEAN_OP, MATRIX_ROW_ST_DEV_OP, VECTOR_ST_DEV_OP, MATRIX_ST_DEV_OP, VECTOR_UNIFORM_SET_OP, MATRIX_UNIFORM_SET_OP, CORRELATION_OP, MATRIX_GET_ROW_OP, MATRIX_GET_COLUMN_OP, COVARIANCE_OP, MATRIX_GET_SCALAR_OP] \
        setup_size_init: 10 \
        mutate_setup_size_min: 10 \
        mutate_setup_size_max: 21 \
        predict_size_init: 50 \
        mutate_predict_size_min: 30 \
        mutate_predict_size_max: 71 \
        learn_size_init: 20 \
        mutate_learn_size_min: 20 \
        mutate_learn_size_max: 45 \
        fec {num_train_examples: 10 num_valid_examples: 10} \
        fitness_combination_mode: MEAN_FITNESS_COMBINATION \
        population_size: 100 \
        tournament_size: 10 \
        initial_population: $init \
        max_train_steps: $steps \
        allowed_mutation_types {
          mutation_types: [ALTER_PARAM_MUTATION_TYPE, RANDOMIZE_COMPONENT_FUNCTION_MUTATION_TYPE, INSERT_INSTRUCTION_MUTATION_TYPE, REMOVE_INSTRUCTION_MUTATION_TYPE, RANDOMIZE_INSTRUCTION_MUTATION_TYPE, TRADE_INSTRUCTION_MUTATION_TYPE, RANDOMIZE_ALGORITHM_MUTATION_TYPE] \
        } \
        mutate_prob: 0.9 \
        progress_every: 10000 \
        " \
      --random_seed=${try_random_seeds:-1000099} \