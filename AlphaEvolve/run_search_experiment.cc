// Copyright 2020 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Modifications copyright (C) 2021 <NUS/Cui Can>

// Runs the RegularizedEvolution algorithm locally.

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include "algorithm.h"
#include "task_util.h"
#include "task.pb.h"
#include "definitions.h"
#include "instruction.pb.h"
#include "evaluator.h"
#include "experiment.pb.h"
#include "experiment_util.h"
#include "fec_cache.h"
#include "generator.h"
#include "mutator.h"
#include "random_generator.h"
#include "regularized_evolution.h"
#include "train_budget.h"
#include "google/protobuf/text_format.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/time/time.h"

typedef alphaevolve::IntegerT IntegerT;
typedef alphaevolve::RandomSeedT RandomSeedT;
typedef alphaevolve::InstructionIndexT InstructionIndexT;

ABSL_FLAG(
    std::string, the_output_path_for_evolved_alphas_and_performance, "",
    "The output path for evolved alphas and performance.");
ABSL_FLAG(
    std::string, my_alpha_file_path, "",
    "The path for my own alpha.");
ABSL_FLAG(
    std::string, cutoff_valid, "",
    "The cutoff validation returns from previous rounds of alpha search.");
ABSL_FLAG(
    std::string, cutoff_test, "",
    "The cutoff test returns from previous rounds of alpha search.");
ABSL_FLAG(
    IntegerT, test_efficiency, 1,
    "If we perform the test for efficiency.");
ABSL_FLAG(
    IntegerT, predict_index, 1,
    "Whether this task is to predict index value.");
ABSL_FLAG(
    double, predict_index_confidence, 0.6,
    "Only larger than this value then hold a long position.");
ABSL_FLAG(
    IntegerT, num_top_stocks, 50,
    "The number of stocks to select in long short portfolio.");
ABSL_FLAG(
    std::string, generate_preds_data, "",
    "The output path for generating alpha outputs as data.");
ABSL_FLAG(
    IntegerT, all_timesteps, 1244,
    "The number of timesteps used in alpha tasks.");
ABSL_FLAG(
    std::string, search_experiment_spec, "",
    "Specification for the experiment. Must be an SearchExperimentSpec proto in text-format. Required.");
ABSL_FLAG(
    std::string, final_tasks, "",
    "The tasks to use for the final evaluation. Must be a TaskCollection "
    "proto in text format. Required.");
ABSL_FLAG(
    IntegerT, max_experiments, 1,
    "Number of experiments to run. The code may end up running fewer "
    "if `sufficient_fitness` is set. If `0`, runs indefinitely.");
ABSL_FLAG(
    RandomSeedT, random_seed, 0,
    "Seed for random generator. Use `0` to not specify a seed (creates a new "
    "seed each time). If running multiple experiments, this seed is set at the "
    "beginning of the first experiment. Does not affect tasks.");
ABSL_FLAG(
    bool, randomize_task_seeds, false,
    "If true, the data in T_search and T_select is randomized for every "
    "experiment (including the first one). That is, any seeds specified in "
    "the search_tasks inside the search_experiment_spec or in the "
    "select_tasks are ignored. (Seeds in final_tasks are still "
    "respected, however).");
ABSL_FLAG(
    std::string, select_tasks, "",
    "The tasks to use in T_select. Must be a TaskCollection proto "
    "in text-format. Required.");
ABSL_FLAG(
    double, sufficient_fitness, std::numeric_limits<double>::max(),
    "Experimentation stops when any experiment reaches this select fitness. "
    "If not specified, keeps experimenting until max_experiments is reached.");

namespace alphaevolve {

namespace {
using ::absl::GetCurrentTimeNanos;  // NOLINT
using ::absl::GetFlag;  // NOLINT
using ::absl::make_unique;  // NOLINT
using ::std::cout;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::mt19937;  // NOLINT
using ::std::numeric_limits;  // NOLINT
using ::std::shared_ptr;  // NOLINT
using ::std::unique_ptr;  // NOLINT
using ::std::vector;  // NOLINT
}  // namespace

void run() {
  // Set random seed.
  RandomSeedT random_seed = GetFlag(FLAGS_random_seed);
  if (random_seed == 0) {
    random_seed = GenerateRandomSeed();
  }
  mt19937 bit_gen(random_seed);
  RandomGenerator rand_gen(&bit_gen);
  cout << "Random seed = " << random_seed << endl;

  // Build reusable search and select structures.
  CHECK(!GetFlag(FLAGS_search_experiment_spec).empty());
  auto experiment_spec = ParseTextFormat<SearchExperimentSpec>(
      GetFlag(FLAGS_search_experiment_spec));
  const double sufficient_fitness = GetFlag(FLAGS_sufficient_fitness);
  const IntegerT max_experiments = GetFlag(FLAGS_max_experiments);

  /// reading from file the init algo
  vector<std::string> alpha_from_file; 
  std::string folder_and_name;
  switch (experiment_spec.initial_population()) { 
    case MY_ALPHA: {
      std::ifstream infile;
      folder_and_name = GetFlag(FLAGS_the_output_path_for_evolved_alphas_and_performance);
      infile.open (GetFlag(FLAGS_my_alpha_file_path));       
      std::string line;
      while (std::getline(infile, line)) {
        alpha_from_file.push_back(line);
      }
      infile.close();
      break;      
    }
    case ALPHA_101: {
      std::ifstream infile;
      folder_and_name = GetFlag(FLAGS_the_output_path_for_evolved_alphas_and_performance);
      infile.open (GetFlag(FLAGS_my_alpha_file_path)); 
      std::string line;
      while (std::getline(infile, line)) {
        alpha_from_file.push_back(line);
      }
      infile.close();
      break;
    }
    default:
      break;
  } 
  folder_and_name = GetFlag(FLAGS_the_output_path_for_evolved_alphas_and_performance); 
  Generator generator(
      experiment_spec.initial_population(),
      experiment_spec.setup_size_init(),
      experiment_spec.predict_size_init(),
      experiment_spec.learn_size_init(),
      ExtractOps(experiment_spec.setup_ops()),
      ExtractOps(experiment_spec.predict_ops()),
      ExtractOps(experiment_spec.learn_ops()), &bit_gen,
      &rand_gen, &alpha_from_file);
  unique_ptr<TrainBudget> train_budget;
  if (experiment_spec.has_train_budget()) {
    train_budget =
        BuildTrainBudget(experiment_spec.train_budget(), &generator);
  }
  Mutator mutator(
      experiment_spec.allowed_mutation_types(),
      experiment_spec.mutate_prob(),
      ExtractOps(experiment_spec.setup_ops()),
      ExtractOps(experiment_spec.predict_ops()),
      ExtractOps(experiment_spec.learn_ops()),
      experiment_spec.mutate_setup_size_min(),
      experiment_spec.mutate_setup_size_max(),
      experiment_spec.mutate_predict_size_min(),
      experiment_spec.mutate_predict_size_max(),
      experiment_spec.mutate_learn_size_min(),
      experiment_spec.mutate_learn_size_max(),
      &bit_gen, &rand_gen);
  auto select_tasks =
      ParseTextFormat<TaskCollection>(GetFlag(FLAGS_select_tasks));
  // Run search experiments and select best algorithm.
  IntegerT num_experiments = 0;
  double best_select_fitness = numeric_limits<double>::lowest();
  shared_ptr<const Algorithm> best_algorithm = make_shared<const Algorithm>();
  while (true) {
    // Randomize T_search tasks.
    if (GetFlag(FLAGS_randomize_task_seeds)) {
      RandomizeTaskSeeds(experiment_spec.mutable_search_tasks(),
                            rand_gen.UniformRandomSeed());
    }
    // Build non-reusable search structures.
    unique_ptr<FECCache> functional_cache =
        experiment_spec.has_fec() ?
            make_unique<FECCache>(experiment_spec.fec()) :
            nullptr;

    Evaluator evaluator(
        experiment_spec.fitness_combination_mode(),
        experiment_spec.search_tasks(),
        &rand_gen, functional_cache.get(), train_budget.get(),
        experiment_spec.max_abs_error(),
        GetFlag(FLAGS_cutoff_valid), GetFlag(FLAGS_cutoff_test), GetFlag(FLAGS_test_efficiency),
        GetFlag(FLAGS_predict_index), GetFlag(FLAGS_predict_index_confidence), GetFlag(FLAGS_num_top_stocks),
        GetFlag(FLAGS_generate_preds_data), GetFlag(FLAGS_all_timesteps));
    RegularizedEvolution regularized_evolution(
        &rand_gen, experiment_spec.population_size(),
        experiment_spec.tournament_size(),
        experiment_spec.progress_every(),
        &generator, &evaluator, &mutator, folder_and_name);
    
    // Run one experiment.
    cout << "Running evolution experiment (on the T_search tasks)..." << endl;
    
    regularized_evolution.Init();
    const IntegerT remaining_train_steps =
        experiment_spec.max_train_steps() -
        regularized_evolution.NumTrainSteps();
    
    regularized_evolution.Run(remaining_train_steps, kUnlimitedTime);
    cout << "Experiment done. Retrieving candidate algorithm." << endl;
    
    // Extract best algorithm based on T_search.
    double unused_pop_mean, unused_pop_stdev, search_fitness, search_IC, unused_pop_sharpe_ratio, unused_pop_correlation, unused_pop_average_holding_days, unused_pop_max_dropdown, 
    unused_pop_strat_ret_vol, unused_pop_annual_mean_strat_ret, unused_pop_correlation_with_existing_alpha, unused_pop_test_sharpe_ratio, unused_pop_test_correlation, unused_pop_test_average_holding_days, 
    unused_pop_test_max_dropdown, unused_pop_test_strat_ret_vol, unused_pop_test_annual_mean_strat_ret, unused_pop_test_correlation_with_existing_alpha;
    shared_ptr<const Algorithm> candidate_algorithm =
        make_shared<const Algorithm>();
    std::vector<double> unused_best_valid_strategy_return, unused_best_test_strategy_return;
    regularized_evolution.PopulationStats(
        &unused_pop_mean, &unused_pop_stdev,
        &candidate_algorithm, &search_fitness, &search_IC, &unused_pop_sharpe_ratio, 
        &unused_pop_correlation, &unused_pop_average_holding_days, &unused_pop_max_dropdown, 
        &unused_pop_strat_ret_vol, &unused_pop_annual_mean_strat_ret, &unused_pop_correlation_with_existing_alpha, &unused_best_valid_strategy_return, &unused_pop_sharpe_ratio, 
        &unused_pop_correlation, &unused_pop_average_holding_days, &unused_pop_max_dropdown, &unused_pop_strat_ret_vol, &unused_pop_annual_mean_strat_ret, &unused_pop_test_correlation_with_existing_alpha, &unused_best_test_strategy_return);
    cout << "Search fitness for candidate algorithm = "
         << search_fitness << endl;
    
    // Randomize T_select tasks.
    if (GetFlag(FLAGS_randomize_task_seeds)) {
      RandomizeTaskSeeds(&select_tasks, rand_gen.UniformRandomSeed());
    }
    mt19937 select_bit_gen(rand_gen.UniformRandomSeed());
    RandomGenerator select_rand_gen(&select_bit_gen);

    // Consider stopping experiments.
    if (sufficient_fitness > 0.0 &&
        best_select_fitness > sufficient_fitness) {
      // Stop if we reached the specified `sufficient_fitness`.
      break;
    }
    ++num_experiments;
    if (max_experiments != 0 && num_experiments >= max_experiments) {
      //Stop if we reached the maximum number of experiments.
      break;
    }
  }

}

}  // namespace alphaevolve

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  alphaevolve::run();
  return 0;
}
