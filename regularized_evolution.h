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

#ifndef REGULARIZED_EVOLUTION_H_
#define REGULARIZED_EVOLUTION_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "algorithm.h"
#include "definitions.h"
#include "evaluator.h"
#include "generator.h"
#include "mutator.h"
#include "random_generator.h"
#include "absl/flags/flag.h"
#include "absl/time/time.h"
#include "gtest/gtest_prod.h"

namespace alphaevolve {

class RegularizedEvolution {
 public:
  RegularizedEvolution(
      // The compute cost of evaluating one individual.
      RandomGenerator* rand_gen,
      // Runs up to this many total individuals.
      IntegerT population_size,
      IntegerT tournament_size,
      // How frequently to print progress reports.
      IntegerT progress_every,
      Generator* generator,
      Evaluator* evaluator,
      // The mutator to use to perform all mutations.
      Mutator* mutator,
      std::string folder_and_name);
  RegularizedEvolution(
      const RegularizedEvolution& other) = delete;
  RegularizedEvolution& operator=(
      const RegularizedEvolution& other) = delete;

  // Initializes the algorithm. Returns the number of individuals evaluated in
  // this call.
  IntegerT Init();

  // Runs for a given amount of time (rounded up to the nearest generation) or
  // for a certain number of train steps (rounded up to the nearest generation),
  // whichever is first. Assumes that Init has been called. Returns the number
  // of train steps executed in this call.
  IntegerT Run(IntegerT max_train_steps, IntegerT max_nanos);

  // Returns the CUs/number of individuals evaluated so far. Returns an exact
  // number.
  IntegerT NumIndividuals() const;

  // The number of train steps executed.
  IntegerT NumTrainSteps() const;

  // Returns a random serialized Algorithm in the population and its fitness.
  std::shared_ptr<const Algorithm> Get(double* fitness);

  // Returns the best serialized Algorithm in the population and its worker
  // fitness.
  std::shared_ptr<const Algorithm> GetBest(double* fitness);

  IntegerT PopulationSize() const;

  void PopulationStats(
      double* pop_mean, double* pop_stdev,
      std::shared_ptr<const Algorithm>* pop_best_algorithm,
      double* pop_best_fitness, double* pop_best_IC, double* pop_best_sharpe_ratio, 
      double* pop_best_correlation, double* pop_best_average_holding_days, double* pop_best_max_dropdown, 
      double* pop_best_strat_ret_vol, double* pop_best_annual_mean_strat_ret, double* pop_best_correlation_with_existing_alpha,
      std::vector<double>* best_valid_strategy_return,
      double* pop_test_sharpe_ratio, double* pop_test_correlation, double* pop_test_average_holding_days, 
      double* pop_test_max_dropdown, double* pop_test_strat_ret_vol, double* pop_test_annual_mean_strat_ret,
      double* pop_test_correlation_with_existing_alpha,
      std::vector<double>* best_test_strategy_return) const;

 private:
  FRIEND_TEST(RegularizedEvolutionTest, TimesCorrectly);

  friend IntegerT PutsInPosition(
      const Algorithm&, RegularizedEvolution*);
  friend IntegerT EvaluatesAndPutsInPosition(
      const Algorithm&, RegularizedEvolution*);
  friend bool PopulationsEq(
      const RegularizedEvolution&,
      const RegularizedEvolution&);

  void InitAlgorithm(std::shared_ptr<const Algorithm>* algorithm);
  std::pair<double, std::vector<double>> Execute(std::shared_ptr<const Algorithm> algorithm, double best_train_fitness, std::vector<double>* ICs, std::vector<double>* strategy_ret, std::vector<double>* valid_strategy_ret, std::vector<IntegerT>* useful_list);
  std::shared_ptr<const Algorithm> BestFitnessTournament();
  void SingleParentSelect(std::shared_ptr<const Algorithm>* algorithm);
  void MaybePrintProgress(std::vector<double>* strategy_ret, std::vector<double>* valid_strategy_ret, std::vector<IntegerT>* useful_list, bool final_result=false);

  Evaluator* evaluator_;
  RandomGenerator* rand_gen_;
  const IntegerT start_secs_;
  IntegerT epoch_secs_;
  IntegerT epoch_secs_last_progress_;
  IntegerT num_individuals_last_progress_;
  const IntegerT tournament_size_;
  const IntegerT progress_every_;
  bool initialized_;
  Generator* generator_;
  Mutator* mutator_;
  std::string folder_and_name_;
  //std::vector<std::vector<double>> all_test_strategy_ret_;
  //std::vector<std::vector<double>> all_valid_strategy_ret_;
  //all_test_strategy_ret_.reserve(population_size_);
  //all_valid_strategy_ret_.reserve(population_size_);

  // Serializable components.
  const IntegerT population_size_;
  //std::vector<std::vector<double>> all_test_strategy_ret_.reserve(population_size_);
  //std::vector<std::vector<double>> all_valid_strategy_ret_.reserve(population_size_);
  std::vector<std::shared_ptr<const Algorithm>> algorithms_;
  std::vector<std::pair<double, std::vector<double>>> fitnesses_;
  std::vector<double> ICs_;
  IntegerT num_individuals_;
  std::vector<std::vector<double>> all_test_strategy_ret_;
  std::vector<std::vector<double>> all_valid_strategy_ret_;
};

}  // namespace alphaevolve

#endif  // REGULARIZED_EVOLUTION_H_
