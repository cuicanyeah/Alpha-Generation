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

while getopts a:p:m:s:v:t:e:i:c:n:d:g:b:f:h:j:o:l: flag
do
    case "${flag}" in
        a) init=${OPTARG};; # select how we initialize our population, e.g., MY_ALPHA to build the initial population using a given alpha sepecified by -m
        p) the_output_path_for_evolved_alphas_and_performance=${OPTARG};; # a directory for outputing alpha performance results
        m) my_alpha_file_path=${OPTARG};; # the directory for initial alpha being evolved
        s) steps=${OPTARG};; # maximum number steps for the evolution
        v) cutoff_valid=${OPTARG};; # a vector of returns on the validation period for weak correlation check
        t) cutoff_test=${OPTARG};; # a vector of returns on the test period for test if there is still weak correlation on test period
        e) test_efficiency=${OPTARG};; # if we are testing efficiency
        i) predict_index=${OPTARG};; # if we are predicting stock index
        c) predict_index_confidence=${OPTARG};; # if we are predicting stock index, then set a confidence interval
        n) num_top_stocks=${OPTARG};; # number of top/bottom stocks in the long short trading strategy
        d) try_random_seeds=${OPTARG};; # random seed used in the evolutionary process
        g) generate_preds_data=${OPTARG};; # a directory for outputing alpha predictions. When we provide a path, we are in evaluation mode and terminate early after evaluation.
        b) stock_market=${OPTARG};; # the stock market we used as dataset [NASDAQ, NYSE, ALL]
        f) num_stocks=${OPTARG};; # the total number of stocks we considered in our stock universe
        h) num_train_samples=${OPTARG};; # num_train_samples should follow the number determined by the generate_datasets.py (i.e., num_train_examples) and specified in DATA_DIR
        j) num_valid_samples=${OPTARG};; # num_valid_samples should follow the number determined by the generate_datasets.py (i.e., num_valid_examples) and specified in DATA_DIR
        o) input_data_folder=${OPTARG};; # the directory for the input data
        l) cache_dir=${OPTARG};; # the directory to hold cache. Should have sufficient space.
        # k) if_evaluate=${OPTARG};; # if it is evaluating the we set sample to the min of 13 and use valid samples for evaluation
    esac
done

all_timesteps=$(( num_train_samples + num_valid_samples ))

DATA_DIR=${input_data_folder} # _preds

bazel --output_user_root=${cache_dir} run -c opt \
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
              dataset_name: '$stock_market' \
              path: '${DATA_DIR}' \
              max_supported_data_seed: 1 \
            } \
            features_size: 13 \
            num_train_examples: ${num_train_samples:-1} \
            num_valid_examples: ${num_valid_samples:-471} \
            num_train_epochs: 1 \
            num_tasks: ${num_stocks:-1402} \
            eval_type: NO_CHANGE\
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
          mutation_types: [ALTER_PARAM_MUTATION_TYPE, RANDOMIZE_COMPONENT_FUNCTION_MUTATION_TYPE, INSERT_INSTRUCTION_MUTATION_TYPE, REMOVE_INSTRUCTION_MUTATION_TYPE] \
        } \
        mutate_prob: 0.9 \
        progress_every: 10000 \
        " \
      --random_seed=${try_random_seeds:-1000099} \
