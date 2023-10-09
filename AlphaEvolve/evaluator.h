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

#ifndef EVALUATOR_H_
#define EVALUATOR_H_

#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "algorithm.h"
#include "task.h"
#include "task.pb.h"
#include "definitions.h"
#include "experiment.pb.h"
#include "fec_cache.h"
#include "random_generator.h"
#include "train_budget.h"

namespace alphaevolve {

class Algorithm;

// See base class.
class Evaluator {
 public:
  Evaluator(
      const FitnessCombinationMode fitness_combination_mode,
      // Tasks to use. Will be filtered to only keep tasks targeted
      // to this worker.
      const TaskCollection& task_collection,
      // The random generator seed to use for any random operations that
      // may be executed by the component function (e.g. VectorRandomInit).
      RandomGenerator* rand_gen,
      // An cache to avoid reevaluating models that are functionally
      // identical. Can be nullptr.
      FECCache* functional_cache,
      // A train budget to use.
      TrainBudget* train_budget,
      // Errors larger than this trigger early stopping, as they signal
      // models that likely have runnaway behavior.
      double max_abs_error,
      // If false, suppresses all logging output. Finer grain control
      // available through logging flags.
      std::string valid_cutoff, 
      std::string test_cutoff,
      IntegerT test_efficiency,
      IntegerT predict_index,
      double predict_index_confidence,
      IntegerT num_top_stocks,
      std::string generate_preds_data,
      IntegerT all_timesteps);

  Evaluator(const Evaluator& other) = delete;
  Evaluator& operator=(const Evaluator& other) = delete;
  
  // Evaluates a Algorithm by executing it on the tasks. Returns the mean
  // fitness.
  // double Evaluate(const Algorithm& algorithm, double best_select_fitness = 0, bool is_train = true, vector<double>& ICs = nullptr);
  std::pair<double, std::vector<double>> Evaluate(const Algorithm& algorithm, double best_select_fitness = 0, IntegerT is_search = 0, std::vector<double>* strategy_ret = nullptr, std::vector<double>* valid_strategy_ret = nullptr, std::vector<IntegerT>* useful_list = nullptr);
  // Get the number of train steps this evaluator has performed.
  IntegerT GetNumTrainStepsCompleted() const;

  std::vector<double> getVertexIndices(std::string const& pointLine);
  
  std::array<std::vector<double>, 6> SplitStringWithDelimiter(std::string s, std::string delimiter);

  double CorrelationWithExisting(std::vector<double>* valid_strategy_ret, std::vector<double> existing_alpha_ret, IntegerT all_task_preds_size, IntegerT existing_alpha_ret_test_size);

  bool CheckHasIn(const Algorithm* algorithm, IntegerT ins_type, IntegerT out, vector<IntegerT>* useful_list, std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>* check_cycle_list, IntegerT main_pos, IntegerT if_predict_only);

  std::vector<std::vector<double> > Transpose(const std::vector<std::vector<double> > data);

  std::vector<std::vector<double> > GetValidTest(const std::vector<std::vector<double> > data, bool is_validate);

  std::vector<IntegerT> TopkLarge(const std::vector<double> test, const int k);

  std::vector<IntegerT> TopkSmall(const std::vector<double> test, const int k);
  
  double ComputeIndexTradingMeasures(const std::vector<std::vector<double> > all_task_preds, const std::vector<std::vector<double> > price_diff, double* sharpe_ratio, double* average_holding_days, double* strat_ret_vol, double* annual_mean_strat_ret, std::vector<double>* strategy_ret, bool is_validate);

  double ComputeAllMeasure(const std::vector<std::vector<double> > all_task_preds, const std::vector<std::vector<double> > price_diff, double* sharpe_ratio, double* average_holding_days, double* strat_ret_vol, double* annual_mean_strat_ret, std::vector<double>* strategy_ret, bool is_validate);

  double CorrelationVec(const std::vector<double> all_task_preds, const std::vector<double> price_diff);

  double Correlation(const std::vector<std::vector<double> > all_task_preds, const std::vector<std::vector<double> > price_diff);

  double stdev(const std::vector<double> v);

  double mean(const std::vector<double> v);

  bool CheckCompleteness(const std::vector<std::vector<double> > data);
 private:
  double Execute(const TaskInterface& task, IntegerT num_train_examples,
                 const Algorithm& algorithm, std::vector<std::vector<double>>* all_task_preds, std::vector<std::vector<double>>* all_price_diff, std::vector<std::vector<std::vector<double>>>* tasks_rank, IntegerT this_stock_to_change, IntegerT task_index, IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, const IntegerT all_rounds, std::vector<IntegerT> *useful_list);

  template <FeatureIndexT F>
  double ExecuteImpl(const Task<F>& task, IntegerT num_train_examples,
                     const Algorithm& algorithm, std::vector<std::vector<double>>* all_task_preds, std::vector<std::vector<double>>* all_price_diff, std::vector<std::vector<std::vector<double>>>* tasks_rank, IntegerT this_stock_to_change, IntegerT task_index, IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, const IntegerT all_rounds, std::vector<IntegerT> *useful_list);

  double CapFitness(double fitness);

  const FitnessCombinationMode fitness_combination_mode_;

  // Contains only task specifications targeted to his worker.
  const TaskCollection task_collection_;

  TrainBudget* train_budget_;
  RandomGenerator* rand_gen_;
  std::vector<std::unique_ptr<TaskInterface>> tasks_;
  FECCache* functional_cache_;
  std::unique_ptr<std::mt19937> functional_cache_bit_gen_owned_;
  std::unique_ptr<RandomGenerator> functional_cache_rand_gen_owned_;
  RandomGenerator* functional_cache_rand_gen_;
  const std::vector<RandomSeedT> first_param_seeds_;
  const std::vector<RandomSeedT> first_data_seeds_;

  double best_fitness_;
  std::shared_ptr<Algorithm> best_algorithm_;

  const double max_abs_error_;
  IntegerT num_train_steps_completed_;
  std::string valid_cutoff_;
  std::string test_cutoff_;
  IntegerT test_efficiency_;
  IntegerT predict_index_;
  double predict_index_confidence_;
  IntegerT num_top_stocks_;
  std::string generate_preds_data_;
  IntegerT all_timesteps_;
};

namespace internal {

double CombineFitnesses(
    const std::vector<double>& task_fitnesses,
    const FitnessCombinationMode mode);

}  // namespace internal

}  // namespace alphaevolve

#endif  // EVALUATOR_H_
