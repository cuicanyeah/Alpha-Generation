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

#include "evaluator.h"

#include <algorithm>
#include <iomanip>
#include <ios>
#include <limits>
#include <memory>
#include <string>
#include <fstream>
#include <queue>
#include <cmath>
#include <stdlib.h> 
#include <sstream>
#include <iterator>

#include "task.h"
#include "task_util.h"
#include "task.pb.h"
#include "definitions.h"
#include "executor.h"
#include "random_generator.h"
#include "train_budget.h"
#include "google/protobuf/text_format.h"
#include "absl/algorithm/container.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

namespace alphaevolve {

bool operatorf(std::pair<double, int> i, std::pair<double, int> j) { return (i.first > j.first);}

using ::absl::c_linear_search;  // NOLINT
using ::absl::GetFlag;  // NOLINT
using ::absl::make_unique;  // NOLINT
using ::std::cout;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::fixed;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::min;  // NOLINT
using ::std::mt19937;  // NOLINT
using ::std::nth_element;  // NOLINT
using ::std::pair;  // NOLINT
using ::std::setprecision;  // NOLINT
using ::std::vector;  // NOLINT
using ::std::unique_ptr;  // NOLINT
using internal::CombineFitnesses;

constexpr IntegerT kMinNumTrainExamples = 1; // previously it was 10 here, I change it to 1 because when generating alpha predictions as feature data, training days are much less
constexpr RandomSeedT kFunctionalCacheRandomSeed = 235732282;

Evaluator::Evaluator(const FitnessCombinationMode fitness_combination_mode,
                     const TaskCollection& task_collection,
                     RandomGenerator* rand_gen,
                     FECCache* functional_cache,
                     TrainBudget* train_budget,
                     const double max_abs_error,
                     std::string valid_cutoff,
                     std::string test_cutoff,
                     IntegerT test_efficiency,
                     IntegerT predict_index,
                     double predict_index_confidence,
                     IntegerT num_top_stocks,
                     std::string generate_preds_data,
                     IntegerT all_timesteps)
    : fitness_combination_mode_(fitness_combination_mode),
      task_collection_(task_collection),
      train_budget_(train_budget),
      rand_gen_(rand_gen),
      functional_cache_(functional_cache),
      functional_cache_bit_gen_owned_(
          make_unique<mt19937>(kFunctionalCacheRandomSeed)),
      functional_cache_rand_gen_owned_(
          make_unique<RandomGenerator>(functional_cache_bit_gen_owned_.get())),
      functional_cache_rand_gen_(functional_cache_rand_gen_owned_.get()),
      best_fitness_(-1.0),
      max_abs_error_(max_abs_error),
      num_train_steps_completed_(0),
      valid_cutoff_(valid_cutoff),
      test_cutoff_(test_cutoff),
      test_efficiency_(test_efficiency),
      predict_index_(predict_index),
      predict_index_confidence_(predict_index_confidence),
      num_top_stocks_(num_top_stocks),
      generate_preds_data_(generate_preds_data),
      all_timesteps_(all_timesteps) {
  FillTasks(task_collection_, &tasks_);
  CHECK_GT(tasks_.size(), 0);
}

std::pair<double, std::vector<double>> Evaluator::Evaluate(const Algorithm& algorithm, double best_select_fitness, IntegerT is_search, std::vector<double>* strategy_ret, std::vector<double>* valid_strategy_ret, std::vector<IntegerT>* useful_list) {
  // Compute the mean fitness across all tasks.
  vector<double> task_fitnesses;
  task_fitnesses.reserve(tasks_.size());
  vector<IntegerT> task_indexes;  
  vector<vector<double>> all_task_preds;
  vector<vector<double>> all_price_diff;  
  IntegerT num_stock_rank = 0;
  IntegerT num_TS_rank = 0;

  IntegerT num_stock_rank_count = 0;
  IntegerT num_TS_rank_count = 0;

  vector<IntegerT> useful_list_predict_only;
  
  // keep a list of useful operators and prune the useless ones
  useful_list->resize(algorithm.predict_.size() + algorithm.learn_.size());
  useful_list_predict_only.resize(algorithm.predict_.size());

  // when checking useful operators, we check backwards if the output operands are used as subsequent input operands. However, sometimes it forms a cycle which stucks the check
  std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>> check_cycle_list;
  std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>> check_cycle_list_predict_only;
  bool result = CheckHasIn(&algorithm, 0, 1, useful_list, &check_cycle_list, -1, 0); // initialize main pos as -1 since if as 0 then will skip first ins
  bool result_predict_only = CheckHasIn(&algorithm, 0, 1, &useful_list_predict_only, &check_cycle_list_predict_only, -1, 1);

    for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list.begin(); i != check_cycle_list.end(); ++i) {
      if ((i->second).second == 0) {
        (i->second).second = 3;
      }      
    }
    for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list.begin(); i != check_cycle_list.end(); ++i) {
      if ((i->second).second == 3) {
        if (CheckHasIn(&algorithm, (i->first).second, (i->first).first, useful_list, &check_cycle_list, (i->second).first + 1, 0))
         (*useful_list)[(i->second).first] = 1; 
      }      
    }

    // below part is to do the same useful operators checking for predict only since for prediction we need this dependency between instructions in predict function
    for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list_predict_only.begin(); i != check_cycle_list_predict_only.end(); ++i) {
      if ((i->second).second == 0) {
        (i->second).second = 3;
      }      
    }
    for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list_predict_only.begin(); i != check_cycle_list_predict_only.end(); ++i) {
      if ((i->second).second == 3) {
        if (CheckHasIn(&algorithm, (i->first).second, (i->first).first, &useful_list_predict_only, &check_cycle_list_predict_only, (i->second).first + 1, 1))
         useful_list_predict_only[(i->second).first] = 1; 
      }      
    }

  IntegerT predict_only_count = 0;
  for(std::vector<IntegerT>::iterator i = useful_list_predict_only.begin(); i != useful_list_predict_only.end(); ++i) {
    if (*i == 1 && (*useful_list)[predict_only_count + algorithm.learn_.size()] == 0) (*useful_list)[predict_only_count + algorithm.learn_.size()] = 1;
    ++predict_only_count;
  }

  if (test_efficiency_ == 1) {
    for (std::vector<IntegerT>::iterator i = useful_list_predict_only.begin(); i != useful_list_predict_only.end(); ++i)      {*i = 1;}
    for (std::vector<IntegerT>::iterator i = useful_list->begin(); i != useful_list->end(); ++i)
      {*i = 1;}
  }
  
  // get the fingerprint for the alpha by converting the pruned operators into numbers
  vector<double> figure_algo_pred;
  vector<double> figure_algo_learn; 

  IntegerT ins_countprint = 0;
  IntegerT char_num_count = 0;
  for (const std::shared_ptr<const Instruction>& instruction :
       algorithm.predict_) {

    if ((*useful_list)[algorithm.learn_.size() + ins_countprint] < 1) {
      ++ins_countprint;
      continue;
    } else {
      for (int i: instruction->ToString()) {
        if (char_num_count % 1 == 0) {
          figure_algo_pred.push_back((double)i);
          ++char_num_count;
        }
      } 
    } 
   ++ins_countprint;
  }
  IntegerT learn_instr_num = 0;
  for (const std::shared_ptr<const Instruction>& instruction :
       algorithm.learn_) {
    if ((*useful_list)[learn_instr_num] < 1) {
      ++learn_instr_num;
      continue;
    } else {
      for (int i: instruction->ToString()) {
        if (char_num_count % 1 == 0) {
          figure_algo_learn.push_back((double)i);
          ++char_num_count;
        }
      } 
    }
    ++learn_instr_num;
  }

  IntegerT ins_counter = 0;
  for (const std::shared_ptr<const Instruction>& instruction :
       algorithm.predict_) {
    if ((*useful_list)[algorithm.learn_.size() + ins_counter] < 1) {
      ++ins_counter;
      continue;
    }
    // count the different types of rank ops
    if (instruction->op_ == 65 || instruction->op_ == 66 || instruction->op_ == 72 || instruction->op_ == 75) {
      if (instruction->op_ == 65 || instruction->op_ == 66 || instruction->op_ == 75) ++num_stock_rank_count;
      if (instruction->op_ == 72) ++num_TS_rank_count;
    } else if (instruction->op_ == 73) {         
      ++num_stock_rank_count;
      ++num_TS_rank_count;
    }
    ++ins_counter;
  }

  double sharpe_ratio;
  double average_holding_days;
  double strat_ret_vol;
  double annual_mean_strat_ret;
  double correlation_with_existing_alpha;

  vector<double> result_vector;
  result_vector.resize(14);
  const IntegerT num_of_stocks_to_approximate_rank = tasks_.size(); 
  const IntegerT all_rounds = num_stock_rank_count + 1; // if we found rank ops for stocks, we need to loop over stocks by one more round. The previous rounds are used to stored alpha values, and the last round to calculate the rank op for stocks.

  // check whether we use AlphaEvolve code to generate alpha's preds data. If yes, we add past year's data to generate predictions. e.g., normally 2013-2017, now for preds on 2016, we use 2012-2016
  if (!generate_preds_data_.empty()) all_timesteps_ = 2244-12; // don't know why 2244 here. Maybe just make it larger so no bug happens in executor.h when calculate rank??
  
  // if we detect rank ops in an alpha, we prepare a tasks_rank 3d tensor to hold related values for rank calculation
  vector<vector<vector<double>>> tasks_rank;
  vector<vector<double>> vec_for_push(num_of_stocks_to_approximate_rank, vector<double>(all_timesteps_-12-12)); ///all_timesteps_-12-12 e.g., 1000 + 244 - 12 (train iterator minus 12) - 12 (valid/test iterator minus 12)
  for (IntegerT i = 0; i < (num_stock_rank_count + num_TS_rank_count); ++i) tasks_rank.push_back(vec_for_push);
  for (IntegerT i = 0; i < (num_stock_rank_count + 1) * tasks_.size(); ++i) {
    task_indexes.push_back(i);
  }

  if (functional_cache_ != nullptr) {

    functional_cache_bit_gen_owned_->seed(kFunctionalCacheRandomSeed);

    const size_t hash = functional_cache_->Hash(
        figure_algo_pred, figure_algo_learn, algorithm.predict_.size() + algorithm.learn_.size(), (num_stock_rank_count + 1) * tasks_.size());

    pair<pair<double, vector<double>>, bool> fitness_and_found = functional_cache_->Find(hash);
    if (fitness_and_found.second) { // if we found this alpha being evaluated is already hashed, we read from cache
      functional_cache_->UpdateOrDie(hash, fitness_and_found.first);
      return fitness_and_found.first;
    } else { // else we evaluate this alpha
      for (IntegerT task_index : task_indexes) {
        IntegerT this_round = (task_index / tasks_.size());

        task_index = task_index % tasks_.size();

        const unique_ptr<TaskInterface>& task = tasks_[task_index];
        CHECK_GE(task->MaxTrainExamples(), kMinNumTrainExamples);
        const IntegerT num_train_examples =
            train_budget_ == nullptr ?
            task->MaxTrainExamples() :
            train_budget_->TrainExamples(algorithm, task->MaxTrainExamples());
        double curr_fitness = -1.0;
        curr_fitness = Execute(*task, num_train_examples, algorithm, &all_task_preds, &all_price_diff, &tasks_rank, this_round, task_index, &num_stock_rank, &num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list); 
      }  
      // delete the previous timesteps missed for ranking because for TSranking the samples for the first few samples are not enough for the time window size
      for (auto i = all_task_preds.begin(); i != all_task_preds.end(); ++i) {
        if (!(*i).empty()) {
          for (IntegerT j = 0; j < (num_TS_rank * 13); ++j) (*i).erase((*i).begin());
        }
      } 
      // delete the previous timesteps missed for ranking
      for (auto i = all_price_diff.begin(); i != all_price_diff.end(); ++i) {
        if (!(*i).empty()) {
          for (IntegerT j = 0; j < (num_TS_rank * 13); ++j) (*i).erase((*i).begin());
        }
      } 
      double combined_fitness;

      if (is_search > 0) {
        // check completeness and return IC to array
        if (CheckCompleteness(all_task_preds) && all_task_preds.size() != 0) {
          vector<vector<double>> valid_period_preds = GetValidTest(all_task_preds, true);
          vector<vector<double>> valid_period_diff = GetValidTest(all_price_diff, true);
          vector<vector<double>> test_period_preds = GetValidTest(all_task_preds, false);
          vector<vector<double>> test_period_diff = GetValidTest(all_price_diff, false);

          double correlation = Correlation(valid_period_preds, valid_period_diff);
          double max_dropdown = ComputeAllMeasure(valid_period_preds, valid_period_diff, &sharpe_ratio, &average_holding_days, &strat_ret_vol, &annual_mean_strat_ret, valid_strategy_ret, true);
          if (predict_index_ == 1) max_dropdown = ComputeIndexTradingMeasures(valid_period_preds, valid_period_diff, &sharpe_ratio, &average_holding_days, &strat_ret_vol, &annual_mean_strat_ret, valid_strategy_ret, true);
          double existing_corre1;
          double existing_corre2;
          double existing_corre3;
          double existing_corre4;
          // we read from the argument to parse the existing alphas' return vectors to generate correlation values
          std::string delimiter = ";";
          std::array<vector<double>, 6> all_cutoff_valid_returns = SplitStringWithDelimiter(valid_cutoff_, delimiter);
          std::array<vector<double>, 6> all_cutoff_test_returns = SplitStringWithDelimiter(test_cutoff_, delimiter);
          //vector<double> existing_alpha_ret = {0.001224,-0.001608,-0.001391,-0.000672,0.000167,0.004311,0.002309,0.000706,0.002827,-0.000202,-0.001828,0.000779,0.001701,-0.002960,-0.002118,-0.000290,0.000997,0.001104,0.000500,0.000987,-0.000345,-0.001451,-0.000304,-0.000354,0.000171,-0.002191,0.006876,-0.000870,-0.001405,-0.000080,0.000818,0.001910,-0.002808,0.005351,0.007376,-0.007819,0.005116,-0.000451,0.000083,-0.006025,-0.001118,-0.000759,-0.000163,0.000819,-0.000431,-0.002831,0.002468,-0.006197,0.000165,0.003467,-0.001596,-0.000673,-0.006541,-0.000549,-0.005515,0.001711,-0.001739,0.006246,-0.001597,0.000007,-0.002548,0.001163,-0.006527,0.000296,0.002426,-0.001292,0.001962,-0.001304,-0.000724,0.001967,-0.000708,0.001031,-0.000681,-0.004833,0.000857,-0.002176,-0.002628,-0.002875,-0.004236,0.002247,-0.002525,-0.005086,-0.007315,-0.002520,0.003511,0.004163,0.003474,0.001585,0.003983,0.001598,-0.003088,-0.002285,0.000868,0.001198,0.002424,0.001169,-0.001485,0.000964,-0.003366,0.002195,-0.000218,-0.002492,0.003624,-0.004654,-0.003878,-0.002047,-0.000415,0.001502,0.004320,-0.000999,-0.003655,-0.000149,-0.001642,-0.008301,-0.005541,-0.002447};
          vector<double> existing_alpha_ret_test = {-0.001599,-0.000963,-0.004110,0.000290,-0.003303,0.006604,-0.000512,-0.004431,0.001947,-0.003298,0.001147,-0.003483,0.001902,-0.002309,0.002586,-0.004156,-0.001108,0.001460,0.002033,-0.005519,-0.003722,-0.003996,-0.001980,0.002942,-0.001990,0.006957,0.001399,-0.006340,-0.000551,-0.000345,0.002708,0.000006,-0.001902,0.001816,0.002528,0.001461,0.001595,0.001221,0.003618,-0.002809,-0.001231,-0.007098,-0.004042,-0.000532,0.003360,0.002274,-0.002574,-0.006822,-0.001745,-0.002829,-0.010261,-0.003414,-0.000613,-0.002542,-0.004989,0.002362,-0.003858,-0.001314,-0.001282,-0.001294,0.001792,-0.002720,-0.002812,-0.003088,-0.006543,0.003307,-0.002025,-0.004060,0.000821,-0.000701,-0.001232,0.000728,0.006457,-0.001944,0.002227,0.004321,0.003057,0.002156,-0.003656,-0.002231,0.000770,-0.004623,0.002438,-0.001151,0.001699,-0.001660,0.000810,-0.000366,-0.001277,-0.000355,0.004241,-0.000744,-0.005438,0.004717,0.000816,-0.009150,-0.006324,0.001075,0.002367,0.003602,-0.004215,-0.007240,-0.001721,0.001578,-0.001861,0.001590,0.003218,-0.002555,-0.006567,-0.001343,-0.000147,-0.003605,0.003136,0.002115,-0.002631,-0.000428};
          
          vector<double> existing_alpha_ret_2 = {-0.001883,-0.004164,-0.006156,-0.000723,-0.000825,-0.002015,-0.000632,-0.003243,-0.003910,0.003308,0.000792,0.002087,-0.003378,0.005663,-0.001174,0.001720,0.001232,0.000757,-0.003863,-0.003073,-0.000633,0.002073,-0.003278,0.000753,0.001118,0.001735,-0.004002,-0.001350,-0.000317,-0.002799,0.003504,-0.003172,0.000608,0.001685,-0.006826,-0.001381,-0.001753,0.004187,-0.004757,-0.004196,0.001598,-0.000740,0.000190,0.001127,0.002685,-0.005228,0.000215,-0.000128,0.000079,-0.002302,0.000538,-0.000879,-0.002178,0.003965,0.002216,0.000141,0.007763,0.003497,0.003843,-0.004704,0.002375,-0.005188,-0.001633,-0.003044,0.000029,0.004647,0.002163,0.001272,0.000245,-0.001880,-0.002365,-0.003260,-0.003719,-0.001339,-0.002481,0.000575,-0.001353,-0.003012,0.000328,0.003821,-0.002624,-0.001443,-0.000475,0.001466,0.000329,-0.000518,0.001251,-0.004228,-0.001035,0.002957,-0.000495,0.007179,-0.000867,0.000040,0.000502,0.000480,-0.004651,-0.004409,-0.002326,-0.006420,-0.000302,0.001331,-0.000605,0.001344,-0.001356,0.004333,0.000163,0.002953,-0.000440};
          vector<double> existing_alpha_ret_2_test = {0.001883,-0.000168,-0.001233,-0.008903,-0.000190,-0.002226,-0.001363,-0.002779,0.000221,-0.001752,-0.002399,0.001524,-0.003077,0.002494,0.000170,0.003929,0.000366,0.004866,0.002436,0.002544,-0.000586,-0.000198,-0.001823,0.003771,-0.000912,0.004233,-0.001814,-0.000543,-0.003658,0.002878,-0.000038,0.000492,0.001439,0.000853,-0.002887,0.000208,0.002399,-0.007973,0.000920,-0.002209,-0.003768,0.001465,-0.002088,0.000704,-0.005493,0.001465,0.002010,-0.000487,0.002145,-0.002397,0.002540,0.000213,-0.004127,0.002831,-0.000071,-0.001030,0.001294,0.002630,0.000194,-0.004437,0.000733,-0.000638,-0.001333,0.001072,0.001584,0.002991,0.001479,0.004084,-0.000618,0.000289,0.003202,0.002302,-0.008450,0.000985,-0.000001,-0.000397,-0.001335,-0.001445,0.005921,0.001771,0.000894,-0.010542,0.000522,0.001130,0.003289,0.000355,0.001103,-0.007805,-0.003135,-0.007460,0.000321,-0.002080,0.001830,0.001601,-0.001003,0.000425,-0.000107,0.000876,-0.003948,-0.003380,0.005800,0.004265,0.008672,-0.004443,0.001586,0.004149,-0.006370,0.005204,-0.005072,-0.005982};

          vector<double> existing_alpha_ret_3 = {0.000000,0.000996,0.005810,0.001938,0.004838,0.001349,0.004129,-0.002328,0.002332,0.001349,0.002789,0.000459,0.004275,0.005610,0.005839,0.010633,0.012400,-0.002561,0.001238,0.004556,0.004205,-0.000561,0.001207,0.002411,-0.001629,0.003842,-0.000276,0.007930,0.003751,0.001098,0.001995,0.007071,0.002794,0.001261,-0.002651,0.008355,-0.002822,-0.000389,0.002424,0.006228,0.000010,0.008488,0.004101,0.000661,0.007039,0.003213,0.005136,0.008039,0.007899,0.001179,0.000104,0.000346,0.003013,0.000127,0.000337,0.000693,0.001151,0.001002,0.004882,0.003844,0.001864,0.000555,-0.001499,-0.000278,-0.000455,0.001472,0.004381,0.003914,0.004423,0.003816,0.001980,0.005225,0.004845,0.000771,0.010243,0.005289,0.002162,0.007169,0.004747,0.005430,0.008685,0.003879,0.009971,0.007706,0.006059,0.006542,0.002298,0.002726,-0.003176,0.004193,0.004447,0.002970,0.001620,0.002125,0.000659,0.000749,0.000369,0.002472,0.002112,0.001695,0.000556,0.001992,0.002230,0.003090,0.004406,0.001973,0.001056,0.000098,0.000571,0.004778,0.006007,-0.000356,0.003619,0.002653,0.003842,0.000524}; 
          vector<double> existing_alpha_ret_3_test = {0.001250,0.002782,-0.000779,0.001649,0.000508,0.002558,0.000138,0.002681,0.004757,0.001635,0.002812,0.002991,0.002732,0.002794,0.006887,0.002794,0.002335,0.001737,0.000870,0.005134,0.004168,0.003142,0.003888,0.002514,0.003374,0.005096,0.008249,0.009558,0.002894,0.007235,0.006962,0.000113,0.000472,0.001790,0.000827,0.002128,-0.001719,0.002756,0.000963,0.003424,0.000215,0.003417,0.001978,0.007216,0.001786,0.004674,0.003526,0.004989,0.002272,0.003282,-0.000553,0.000558,0.005754,0.003677,0.003600,0.002926,0.003408,0.005909,0.002166,0.002074,0.003434,0.006721,0.003634,0.004644,0.009817,0.005543,0.001918,0.003861,0.001406,0.002334,0.000531,0.002973,-0.000970,-0.001851,0.002705,0.001924,0.001868,-0.000643,0.004210,0.002784,0.004263,0.003749,0.005839,0.004605,-0.001560,0.008045,0.008386,0.004860,0.002668,0.006251,0.002669,0.013193,0.003009,0.003011,0.007103,0.006148,0.003307,0.002379,0.003067,0.006108,0.001914,0.003999,0.004814,0.002551,0.003416,-0.001254,0.003909,0.005212,0.018763,-0.000817,0.008274,0.010592,0.000859,0.003324,0.001219,-0.000517};

          vector<double> existing_alpha_ret_4 = {0.0040756, 0.0031563, 0.000129051, 0.00373519, -0.000202312, 0.00158072, 0.00538197, 0.00100945, 0.00332539, 0.0024705, 0.00210515, 0.00295206, 0.002621, -0.000790793, -0.00168755, 0.00751675, 0.00288368, 0.00311604, 0.000808149, 0.00377772, -0.00352965, -0.00163943, -0.00455195, 0.00208286, -0.0034128, 0.00221667, 0.00245747, 0.00551712, 0.00368253, 0.00104424, 0.00118564, 0.00359743, -0.00273357, 0.00226265, 0.00603837, 0.00218229, 0.00289919, -0.00300532, 0.0029417, 0.00119422, 0.00062219, 0.000247094, 0.00411388, -0.000347996, 0.000822655, 0.000743444, -0.000333629, 0.00672859, 0.000540362, 0.0024341, 5.27486e-05, 0.00262916, 0.00499986, 0.000204389, -0.0027676, -0.000453751, -9.19366e-05, 0.00223643, 0.000648946, 0.00183285, -0.0010633, 0.00027096, 0.000891729, -0.000132679, 0.00329658, 0.00219543, -0.000815209, 0.00031807, -0.00380936, -0.000548591, -0.000909178, -0.00219513, 0.00101414, 0.00100196, 0.00239809, 0.00233066, 0.0011846, -0.000552008, 0.00217169, -0.00205833, 0.0025243, 0.00350743, 0.00151811, 5.77816e-05, 0.00293952, 0.000792696, 0.000330212, 0.00399359, 0.00134878, -0.00127587, 0.00186386, 0.00345011, 0.00215696, 0.00216855, -0.000666104, -0.000620272, 0.00248596, 0.00236709, 0.00056133, 0.0025395, 0.000756568, 0.00383931, -0.00126461, 0.00139459, -0.00127253, 0.00107539, 0.00159618, -0.00149992, 0.00010252, 0.0010096, 0.000521808, 0.00230177, 0.000694147, -0.000348349, -0.0022005, -0.000207427};
          vector<double> existing_alpha_ret_4_test = {0.00119999, 0.000301742, 0.00302722, 0.00173234, 0.00111756, -0.00218288, -0.00101641, -0.000318455, 0.0024683, 0.00661973, 0.000850705, 0.0010818, -0.00235552, 0.0027329, -0.0021186, 0.00014327, 0.00468092, 0.000622064, 0.00183986, 0.0011201, 0.000779347, 0.00179028, 0.0065174, 0.00107612, 0.000783914, -0.00430983, 0.00182168, 0.000998737, 0.00162662, 0.000603166, -0.000575616, 0.00130931, -0.000291927, -0.00044126, 0.00335834, -0.00236152, -3.48376e-05, 0.00165223, -0.00050383, -0.00120481, 0.00346838, 0.00151728, 0.00228353, 0.000352406, 0.00211582, -0.000921195, -0.0010169, -0.000187047, 0.00548542, 0.00108787, 0.0124939, 0.00214241, -0.000187598, -2.5773e-05, 0.00510192, 0.003419, 0.000525343, 0.00141914, -0.000610481, -0.00058619, -0.00244232, -0.000129933, 0.00309982, 0.000550199, -0.00137139, -0.000291977, -0.00234732, -0.000692767, 0.00292385, 0.0027203, 0.000373473, -0.000782622, -0.000250607, 0.000295233, 0.000224251, 0.000489861, -0.00317021, -0.00164023, 0.000833466, 0.00236122, -8.35385e-05, 0.00208497, 3.83559e-05, 0.000550012, -0.00454016, 0.000271052, -0.00170992, 0.00303823, -0.0024213, 0.00168468, 0.00100232, 0.00384892, 0.00326069, -0.00235716, 0.000161643, 0.00911055, 0.000332738, -0.000423506, -0.00178251, -0.000635346, -0.00215373, 0.00562947, 0.00171501, -0.00949365, 0.00268461, -0.00253421, 0.00199781, 0.00129538, 0.00290936, 0.00239678, 0.0015036, 0.00709739, -0.00208643, -0.000887032, 0.000533674, 0.00168763};
          double corr_sum = 0;
          int num_of_cutoffs = 0;
          for (int x = 0; x < all_cutoff_valid_returns.size(); ++x) {
            if (!all_cutoff_valid_returns[x].empty()) {
            double existing_ith_corr = CorrelationWithExisting(valid_strategy_ret, all_cutoff_valid_returns[x], all_task_preds[0].size(), all_cutoff_test_returns[x].size());
	          if (existing_ith_corr > 0.15) correlation=0;
            corr_sum+=existing_ith_corr;
            num_of_cutoffs++;
            }
          }
          correlation_with_existing_alpha = corr_sum / num_of_cutoffs;// 
    
          result_vector[0] = correlation;
          result_vector[1] = sharpe_ratio;
          result_vector[2] = average_holding_days;
          result_vector[3] = 1 - max_dropdown;
          result_vector[4] = strat_ret_vol;
          result_vector[5] = annual_mean_strat_ret;
          result_vector[6] = correlation_with_existing_alpha;
          // std::cout << "valid sr" << sharpe_ratio << std::endl;
          // std::cout << "valid ahd" << average_holding_days << std::endl; 
          // std::cout << "valid vol" << strat_ret_vol << std::endl; 
          // std::cout << "valid strat ret"<< annual_mean_strat_ret << std::endl;
          combined_fitness = correlation;

          // test part
          double test_correlation = Correlation(test_period_preds, test_period_diff);
          double test_max_dropdown = ComputeAllMeasure(test_period_preds, test_period_diff, &sharpe_ratio, &average_holding_days, &strat_ret_vol, &annual_mean_strat_ret, strategy_ret, false);
          if (predict_index_ == 1) test_max_dropdown = ComputeIndexTradingMeasures(test_period_preds, test_period_diff, &sharpe_ratio, &average_holding_days, &strat_ret_vol, &annual_mean_strat_ret, strategy_ret, false);
          // std::cout << sharpe_ratio << std::endl;
          // std::cout << average_holding_days << std::endl; 
          // std::cout << strat_ret_vol << std::endl; 
          // std::cout << annual_mean_strat_ret << std::endl;
          double test_corr_sum = 0;
          int num_of_test_cutoffs = 0;
          for (int x = 0; x < all_cutoff_test_returns.size(); ++x) {
            if (!all_cutoff_test_returns[x].empty()) {
            double existing_ith_corr = CorrelationVec(all_cutoff_test_returns[x], *strategy_ret); 
            test_corr_sum+=existing_ith_corr;
            num_of_test_cutoffs++;
            }
          } 
          correlation_with_existing_alpha = test_corr_sum / num_of_test_cutoffs;//  
          // cout << "test sharpe_ratio: " << sharpe_ratio << "; test correlation: " << test_correlation << "; test max_dropdown: " << 1 - test_max_dropdown << " correlation_with_existing_alpha: " << correlation_with_existing_alpha << std::endl;
          result_vector[7] = test_correlation;
          result_vector[8] = sharpe_ratio;
          result_vector[9] = average_holding_days;
          result_vector[10] = 1 - test_max_dropdown;
          result_vector[11] = strat_ret_vol;
          result_vector[12] = annual_mean_strat_ret;
          result_vector[13] = correlation_with_existing_alpha;
          
          // check whether we use AlphaEvolve code to generate alpha's preds data (i.e., the predictions of formulaic alphas as features for later training purpose). If yes, save those preds as files
          if (!generate_preds_data_.empty()) {
            std::string filename_returns = generate_preds_data_ + "preds.txt";  
            std::ofstream outFilepreds(filename_returns);
            for (std::vector<double>::size_type i = 0; i < all_task_preds.size(); i++) 
            {
              outFilepreds << "\n";
              for (std::vector<double>::size_type j = 0; j < all_task_preds[i].size(); j++) {
                    outFilepreds << all_task_preds[i][j] << " ";
              }
            }    
            outFilepreds.close();
            std::string filename_labels = generate_preds_data_ + "labels.txt";  
            std::ofstream outFilelabels(filename_labels);
            for (std::vector<double>::size_type i = 0; i < all_price_diff.size(); i++) 
            {
              outFilelabels << "\n ";
              for (std::vector<double>::size_type j = 0; j < all_price_diff[i].size(); j++) {
                    outFilelabels << all_price_diff[i][j] << " ";
              }
            }    
            outFilelabels.close();
            exit( 3 );            
          }

        } else {
          combined_fitness = kMinFitness;
        }
        functional_cache_->InsertOrDie(hash, std::make_pair(combined_fitness, result_vector));
        return std::make_pair(combined_fitness, result_vector);
      }
    }
  }
}

std::vector<double> Evaluator::getVertexIndices(std::string const& pointLine)
{
  std::istringstream iss(pointLine);

  return std::vector<double>{ 
    std::istream_iterator<double>(iss),
    std::istream_iterator<double>()
  };
}

std::array<std::vector<double>, 6> Evaluator::SplitStringWithDelimiter(std::string s, std::string delimiter)
{
  std::array<std::vector<double>, 6> results;
  size_t pos = 0;
  std::string token;
  int i = 0;
  while ((pos = s.find(delimiter)) != std::string::npos) {
	token = s.substr(0, pos);
      	std::vector<double> return_vector = getVertexIndices(token);
        for (int j=0; j<return_vector.size(); j++) 
        {
            results[i].push_back(return_vector[j]);
        }
        i++;
	s.erase(0, pos + delimiter.length());
  }
  
  return results;
}

double Evaluator::CorrelationWithExisting(std::vector<double>* valid_strategy_ret, std::vector<double> existing_alpha_ret, IntegerT all_task_preds_size, IntegerT existing_alpha_ret_test_size) {
  IntegerT length = std::min(valid_strategy_ret->size(), existing_alpha_ret.size());

  if (valid_strategy_ret->size() > existing_alpha_ret.size()) {
    vector<double>::const_iterator first_existing_alpha_ret = existing_alpha_ret.begin();
    vector<double>::const_iterator last_existing_alpha_ret = existing_alpha_ret.begin() + (valid_strategy_ret->size() - (all_task_preds_size - existing_alpha_ret.size() - existing_alpha_ret_test_size));
    vector<double> new_existing_alpha_ret_valid(first_existing_alpha_ret, last_existing_alpha_ret);   

    double correlation_with_existing_alpha = CorrelationVec(new_existing_alpha_ret_valid, *valid_strategy_ret); 
    return correlation_with_existing_alpha;
   
  } else {
    vector<double>::const_iterator first_valid_strategy_ret = valid_strategy_ret->begin();
    vector<double>::const_iterator last_valid_strategy_ret = valid_strategy_ret->begin() + (existing_alpha_ret.size() - (existing_alpha_ret.size() + existing_alpha_ret_test_size - all_task_preds_size));
    vector<double> new_valid_strategy_ret_valid(first_valid_strategy_ret, last_valid_strategy_ret);
    double correlation_with_existing_alpha = CorrelationVec(existing_alpha_ret, new_valid_strategy_ret_valid); 
    return correlation_with_existing_alpha;
  }
}

/// check if previous instruction.out_ has the op previous_rank's instruction.in1_
bool Evaluator::CheckHasIn(const Algorithm* algorithm, IntegerT ins_type, IntegerT out, vector<IntegerT>* useful_list, std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>* check_cycle_list, IntegerT main_pos, IntegerT if_predict_only) {
  vector<double> list_int_op = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19,29,44,47,65,66,72,73,75};
  vector<double> list_int_op2 = {1,2,3,4,44,47};
  vector<double> list_int_op_out = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,21,27,34,44,47,50,51,54,55,56,59,62,65,66,67,71,72,74,75};

  vector<double> list_vec_op = {16,20,21,22,23,24,25,26,27,28,32,33,45,48,50,54,71,74};
  vector<double> list_vec_op2 = {18,23,24,25,26,27,28,31,45,48,71,74};
  vector<double> list_vec_op_out = {16,18,19,20,22,23,24,25,26,31,35,36,45,48,52,53,57,60,63,68,69,73};

  vector<double> list_mat_op = {17,30,31,34,35,36,37,38,39,40,41,42,43,46,49,51,52,53,55};
  vector<double> list_mat_op2 = {29,39,40,41,42,43,46,49};
  vector<double> list_mat_op_out = {17,28,29,30,32,33,37,38,39,40,41,42,43,46,49,58,61,64};

  vector<IntegerT> assign_ops = {56,57,58,59,60,61,62,63,64};
  vector<IntegerT> valid_ops = {67,68,69};

  std::vector<std::shared_ptr<const Instruction>> predict_plus_learn;

  // insert .learn_ first because instructions in predict_ is nearest to s1
  if (if_predict_only == 0) predict_plus_learn.insert( predict_plus_learn.end(), algorithm->learn_.begin(), algorithm->learn_.end() );
  predict_plus_learn.insert( predict_plus_learn.end(), algorithm->predict_.begin(), algorithm->predict_.end() );

  IntegerT pos = -1;
  IntegerT ins_count_find = 0;

  if (check_cycle_list->empty()) {
    // find the position of last s1
    for (const std::shared_ptr<const Instruction>& myinstruction :
     predict_plus_learn) {    
      bool found3 = (std::find(list_int_op_out.begin(), list_int_op_out.end(), myinstruction->op_) != list_int_op_out.end());
      if (found3 && myinstruction->out_ == 1) {
        pos = ins_count_find;
      }
      ++ins_count_find;
    }     
  } else {
    for (const std::shared_ptr<const Instruction>& myinstruction :
     predict_plus_learn) {    
      if (ins_count_find == main_pos) {
        ++ins_count_find;
        continue;
      }

      bool found3 = (std::find(list_int_op_out.begin(), list_int_op_out.end(), myinstruction->op_) != list_int_op_out.end());
      bool found3_vec = (std::find(list_vec_op_out.begin(), list_vec_op_out.end(), myinstruction->op_) != list_vec_op_out.end());
      bool found3_mat = (std::find(list_mat_op_out.begin(), list_mat_op_out.end(), myinstruction->op_) != list_mat_op_out.end());

      switch (ins_type) {
        case 0: {
          if (found3 && myinstruction->out_ == out) {

            if (pos == -1) {
              pos = ins_count_find;             
            }

            else if (((0 < (main_pos - ins_count_find)) && ((main_pos - pos) > (main_pos - ins_count_find))) || ((0 > (main_pos - ins_count_find)) && (0 > (main_pos - pos)) && ((main_pos - pos) > (main_pos - ins_count_find)))) {
            pos = ins_count_find;                          
            }
          }
          break;        
        }
        case 1: {
          if (found3_vec && myinstruction->out_ == out) {
            if (pos == -1) {   
              pos = ins_count_find;            
            }
            else if (((0 < (main_pos - ins_count_find)) && ((main_pos - pos) > (main_pos - ins_count_find))) || ((0 > (main_pos - ins_count_find)) && (0 > (main_pos - pos)) && ((main_pos - pos) > (main_pos - ins_count_find)))) {
              pos = ins_count_find;
            }
          }
          break;        
        }
        case 2: {
          if (found3_mat && myinstruction->out_ == out) {
            if (pos == -1) {
              pos = ins_count_find;
            }           
            else if (((0 < (main_pos - ins_count_find)) && ((main_pos - pos) > (main_pos - ins_count_find))) || ((0 > (main_pos - ins_count_find)) && (0 > (main_pos - pos)) && ((main_pos - pos) > (main_pos - ins_count_find)))) {
              pos = ins_count_find; 
            }
          }   
          break;      
        }
      }

      ++ins_count_find;
    }     

    // this check is including iteration check because iteration will be included into the check cycle list if pos is found other than -1
    for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
      if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second == 0) {
        return false;
      } else if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second == 2) {
        return false;
      }      
    }
  }
  
  // all below code in this function to get the in1 or in2 of this found instruction. And further check if they are output of other instruction  
  if (pos != -1) {
    const std::shared_ptr<const Instruction>& myinstruction = predict_plus_learn[pos];  

    for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
      if ((i->first).first == myinstruction->out_ && (i->first).second == ins_type && (i->second).first == pos && (i->second).second == 1) {
        return true;
      }      
    }

    bool found_assign = (std::find(assign_ops.begin(), assign_ops.end(), myinstruction->op_) != assign_ops.end());
    bool found_valid = (std::find(valid_ops.begin(), valid_ops.end(), myinstruction->op_) != valid_ops.end());

    if (found_assign) (*useful_list)[pos] = 1;
    else if (found_valid) {
      check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1)));
      (*useful_list)[pos] = 1;
      return true;
    } 
    else {
      // assign to unsure 2 first. this is avoid cycle in later check of 1 or 0.
      bool has_cycle_check2 = false;
      for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
        if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 2) {
          i->second.second = 2;
          has_cycle_check2 = true;
        }     
      }

      if (!has_cycle_check2) {
        check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 2))); 
      }

      bool found1 = (std::find(list_int_op.begin(), list_int_op.end(), myinstruction->op_) != list_int_op.end());
      bool found2 = (std::find(list_int_op2.begin(), list_int_op2.end(), myinstruction->op_) != list_int_op2.end());  
      bool found3 = (std::find(list_vec_op.begin(), list_vec_op.end(), myinstruction->op_) != list_vec_op.end());
      bool found4 = (std::find(list_vec_op2.begin(), list_vec_op2.end(), myinstruction->op_) != list_vec_op2.end());  
      bool found5 = (std::find(list_mat_op.begin(), list_mat_op.end(), myinstruction->op_) != list_mat_op.end());
      bool found6 = (std::find(list_mat_op2.begin(), list_mat_op2.end(), myinstruction->op_) != list_mat_op2.end()); 

      if (found2 && found1) {

        bool valid_op_1;

        if (myinstruction->in1_ == 0) valid_op_1 = true;
        else valid_op_1 = CheckHasIn(algorithm, 0, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);

        bool valid_op_2;

        if (myinstruction->in2_ == 0) valid_op_2 = true;
        else valid_op_2 = CheckHasIn(algorithm, 0, myinstruction->in2_, useful_list, check_cycle_list, pos, if_predict_only);
        if (!valid_op_2 && !valid_op_1) {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
          }

          (*useful_list)[pos] = 1;
          return true;        
        } 
      } else if (found4 && found1) {

        bool valid_op_1;

        if (myinstruction->in1_ == 0) valid_op_1 = true;
        else valid_op_1 = CheckHasIn(algorithm, 0, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);

        bool valid_op_2;

        valid_op_2 = CheckHasIn(algorithm, 1, myinstruction->in2_, useful_list, check_cycle_list, pos, if_predict_only);
        if (!valid_op_2 && !valid_op_1) {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
          }

          (*useful_list)[pos] = 1;
          return true;        
        }  
      } else if (found6 && found1) {

        bool valid_op_1;

        if (myinstruction->in1_ == 0) valid_op_1 = true;
        else valid_op_1 = CheckHasIn(algorithm, 0, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);

        bool valid_op_2;

        if (myinstruction->in2_ == 0) valid_op_2 = true;
        else valid_op_2 = CheckHasIn(algorithm, 2, myinstruction->in2_, useful_list, check_cycle_list, pos, if_predict_only);

        if (!valid_op_2 && !valid_op_1) {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
          }

          (*useful_list)[pos] = 1;
          return true;        
        }  
      } else if (found3 && found4) {

        bool valid_op_1;

        valid_op_1 = CheckHasIn(algorithm, 1, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);

        bool valid_op_2;

        valid_op_2 = CheckHasIn(algorithm, 1, myinstruction->in2_, useful_list, check_cycle_list, pos, if_predict_only);
        if (!valid_op_2 && !valid_op_1) {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
          }

          (*useful_list)[pos] = 1;
          return true;        
        }  
      } else if (found5 && found6) {

        bool valid_op_1;

        if (myinstruction->in1_ == 0) valid_op_1 = true;
        else valid_op_1 = CheckHasIn(algorithm, 2, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);

        bool valid_op_2;

        if (myinstruction->in2_ == 0) valid_op_2 = true;
        else valid_op_2 = CheckHasIn(algorithm, 2, myinstruction->in2_, useful_list, check_cycle_list, pos, if_predict_only);

        if (!valid_op_2 && !valid_op_1) {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
          }

          (*useful_list)[pos] = 1;
          return true;        
        }  
      } else if (found5 && found4) {
        bool valid_op_1;

        if (myinstruction->in1_ == 0) valid_op_1 = true;
        else valid_op_1 = CheckHasIn(algorithm, 2, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);

        bool valid_op_2;

        valid_op_2 = CheckHasIn(algorithm, 1, myinstruction->in2_, useful_list, check_cycle_list, pos, if_predict_only);
        if (!valid_op_2 && !valid_op_1) {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
          }

          (*useful_list)[pos] = 1;
          return true;        
        } 
      } else if (found1 && !found2 && !found3 && !found4 && !found5 && !found6) {
        bool valid_op_1;

        if (myinstruction->in1_ == 0) valid_op_1 = true;
        else valid_op_1 = CheckHasIn(algorithm, 0, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);
        if (!valid_op_1) {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
          }

          (*useful_list)[pos] = 1;
          return true;        
        } 
      } else if (!found1 && !found2 && found3 && !found4 && !found5 && !found6) {
        bool valid_op_1;
        valid_op_1 = CheckHasIn(algorithm, 1, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);
        if (!valid_op_1) {
          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {

          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
          }

          (*useful_list)[pos] = 1;
          return true;        
        }       
      } else if (!found1 && !found2 && !found3 && !found4 && found5 && !found6) {
        bool valid_op_1;     
        if (myinstruction->in1_ == 0) valid_op_1 = true; 
        else valid_op_1 = CheckHasIn(algorithm, 2, myinstruction->in1_, useful_list, check_cycle_list, pos, if_predict_only);
        if (!valid_op_1) {
          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 0) {
              i->second.second = 0;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 0))); 
          }

          (*useful_list)[pos] = 0;
          return false;
        } else {
          bool has_cycle = false;
          for(std::vector<std::pair<std::pair<IntegerT, IntegerT>, std::pair<IntegerT, IntegerT>>>::iterator i = check_cycle_list->begin(); i != check_cycle_list->end(); ++i) {
            if ((i->first).first == out && (i->first).second == ins_type && (i->second).first == pos && (i->second).second != 1) {
              i->second.second = 1;
              has_cycle = true;
            }     
          }

          if (!has_cycle) {
            check_cycle_list->push_back(std::make_pair(std::make_pair(out, ins_type), std::make_pair(pos, 1))); 
          }

          (*useful_list)[pos] = 1;
          return true;        
        }
      }
    }   
  } else {
      return false;
  } 
}

bool Evaluator::CheckCompleteness(const std::vector<std::vector<double> > data) {
  for (auto i = data.begin(); i != data.end(); ++i) {
    if ((*i).size() == 0) { 
      return false;
    }
  }
  return true;
}
std::vector<std::vector<double> > Evaluator::Transpose(const std::vector<std::vector<double> > data) {

    std::vector<std::vector<double> > result(data[0].size(),
                                          std::vector<double>(data.size()));

    for (std::vector<double>::size_type i = 0; i < data[0].size(); i++) 
        for (std::vector<double>::size_type j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;
}

std::vector<std::vector<double> > Evaluator::GetValidTest(const std::vector<std::vector<double> > data, bool is_validate) {
  int index =  static_cast<int>(data[0].size()/2);
  std::vector<std::vector<double> > result(data.size(),
                                        std::vector<double>(index));
  std::vector<std::vector<double> > result_not_validate(data.size(),
                                        std::vector<double>(data[0].size() - index));
  if (is_validate) {
    for (std::vector<double>::size_type i = 0; i < data.size(); i++)
      {
        for (std::vector<double>::size_type j = 0; j < index; j++) {
          result[i][j] = data[i][j];
        }
      }
      return result;      
  } else {
    for (std::vector<double>::size_type i = 0; i < data.size(); i++)
      {
        for (std::vector<double>::size_type j = index; j < data[0].size(); j++) {
          result_not_validate[i][j-index] = data[i][j];
        }
      }
      return result_not_validate;      
  }
}

struct myclasslarge {
  bool operator() (std::pair<double, int> &i, std::pair<double, int> &j) { return (i.first > j.first);} /// if all preds are equal then don't change sequence
} myobjectlarge;

std::vector<IntegerT> Evaluator::TopkLarge(const std::vector<double> test, const int k) {
  std::vector<std::pair<double, int>> q;
  std::vector<IntegerT> result;
  for (int i = 0; i < test.size(); ++i) {
    q.push_back(std::pair<double, int>(test[i], i));
  }
  std::stable_sort(q.begin(), q.end(), alphaevolve::operatorf);
  for (int i = 0; i < k; ++i) {
    int ki = q[i].second;
    result.push_back(ki);
  }
  return result;
}

struct myclass {
  bool operator() (std::pair<double, int> &i, std::pair<double, int> &j) { return (i.first < j.first);}
} myobject;

std::vector<IntegerT> Evaluator::TopkSmall(const std::vector<double> test, const int k) {
  std::vector<std::pair<double, int>> q;
  std::vector<IntegerT> result;
  for (int i = 0; i < test.size(); ++i) {
    q.push_back(std::pair<double, int>(test[i], i));
  }
  std::stable_sort(q.begin(), q.end());
  for (int i = 0; i < k; ++i) {
    int ki = q[i].second;
    result.push_back(ki);
  }
  return result;
}

double Evaluator::ComputeIndexTradingMeasures(const std::vector<std::vector<double> > all_task_preds, const std::vector<std::vector<double> > price_diff, double* sharpe_ratio, double* average_holding_days, double* strat_ret_vol, double* annual_mean_strat_ret, std::vector<double>* strategy_ret, bool is_validate) {
    CHECK(all_task_preds.size() == price_diff.size());
    for (std::vector<double>::size_type j = 0; j < all_task_preds.size(); j++) {
      CHECK(all_task_preds[j].size() == price_diff[j].size());
    }
    std::vector<double> preds_column;
    std::vector<double> price_diff_column;
    std::vector<double> strategy_returns;
    double overall_strategy_return = 0.0; 
    double dropdown_strat_ret_accu = 1.0;
    double lowest_dropdown_strat_ret_accu = 1.0;
    double previous_pos = -1.0;
    double current_pos = -1.0;
    IntegerT holding_days_total = 0;
    IntegerT holding_days = 0;
    IntegerT holding_counts = 0;    

    std::vector<double> index_predictions;
    // loop over column and then over rows 
    for (std::vector<double>::size_type i = 0; i < all_task_preds[0].size(); i++) 
      {
        double count_pos_preds = 0.0;
        for (std::vector<double>::size_type j = 0; j < all_task_preds.size()-2; j++) {
          if (std::abs(all_task_preds[j][i]) != 1234 && std::abs(price_diff[j][i]) != 1234) {
            if (all_task_preds[j][i] > 0) {
              count_pos_preds = count_pos_preds + 1.0;
            }
          } 
        }
        overall_strategy_return = 0.0;

        // A random check that the label for the day is the same since it's index prediction
        double trade_rule = count_pos_preds / static_cast<double>(all_task_preds.size()-2);      

        if (all_task_preds[23][i] < (1 - predict_index_confidence_)) {
          current_pos = 2;
          if (std::abs(all_task_preds[23][i]) != 1234 && std::abs(price_diff[23][i]) != 1234) {
            overall_strategy_return = 0 - price_diff[23][i];
          }
        }
        else if (all_task_preds[23][i] > (predict_index_confidence_)) {
          current_pos = 1;
          if (std::abs(all_task_preds[23][i]) != 1234 && std::abs(price_diff[23][i]) != 1234) {
            overall_strategy_return = price_diff[23][i];
          }
        }  
        else 
        {
          current_pos = 0;
          overall_strategy_return = 0;
        }

        if (current_pos == previous_pos && current_pos != 0 && current_pos != -1) holding_days++;
        else if (current_pos != 0 && current_pos != -1) {
          holding_days_total = holding_days_total + holding_days;
          holding_days = 1;
          holding_counts++;
        }
        else if (current_pos != -1) {
          holding_days_total = holding_days_total + holding_days;
          holding_days = 0;    
        }
        previous_pos = current_pos;

        strategy_returns.push_back(overall_strategy_return);

        if (strategy_ret != nullptr) {
          strategy_ret->push_back(overall_strategy_return);
        }

        // Compute the max dropdown
        if (overall_strategy_return < 0) {
          dropdown_strat_ret_accu *= (overall_strategy_return + 1);
          if (lowest_dropdown_strat_ret_accu > dropdown_strat_ret_accu) lowest_dropdown_strat_ret_accu = dropdown_strat_ret_accu;
        } else {
          dropdown_strat_ret_accu = 1.0;
        } 
      }
      *strat_ret_vol = stdev(strategy_returns);
      *annual_mean_strat_ret = mean(strategy_returns) * 252;

      for (std::vector<double>::size_type i = 0; i < strategy_returns.size(); i++) 
      {
        if (strategy_returns[i] > 1) std::cout << "ith strategy_returns >1 !!!!!!!!!!!!!!!!" << strategy_returns[i] << std::endl;
      }

      if (stdev(strategy_returns) == 0 || std::abs(mean(strategy_returns)) < 0.000000001) { /// second condition add because some sharpe ratio is out without valid prediction is due to rounding error, e.g. e-15 / e-16 give a good sharpe ratio
        *sharpe_ratio = 0;
      } else {
        *sharpe_ratio = mean(strategy_returns) / stdev(strategy_returns) * sqrt(252);
      }
      if (holding_counts == 0) {
        *average_holding_days = 0;
      } else {
        holding_days_total = holding_days_total + holding_days; // the last ending case;
        *average_holding_days = holding_days_total / holding_counts;
      }
    return lowest_dropdown_strat_ret_accu;
}

// Helper function to compute Pearson correlation
double pearson_correlation(const std::vector<double>& x, const std::vector<double>& y) {
    double mean_x = 0.0;
    double mean_y = 0.0;
    int valid_count = 0;
    
    for (size_t i = 0; i < x.size(); i++) {
        if (std::abs(x[i]) != 1234 && std::abs(y[i]) != 1234) {
            mean_x += x[i];
            mean_y += y[i];
            valid_count++;
        }
    }
    
    mean_x /= valid_count;
    mean_y /= valid_count;

    double num = 0.0, den_x = 0.0, den_y = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        if (std::abs(x[i]) != 1234 && std::abs(y[i]) != 1234) {
            num += (x[i] - mean_x) * (y[i] - mean_y);
            den_x += std::pow(x[i] - mean_x, 2);
            den_y += std::pow(y[i] - mean_y, 2);
        }
    }
    
    return num / (std::sqrt(den_x) * std::sqrt(den_y));
}

double Evaluator::ComputeAllMeasure(const std::vector<std::vector<double>> all_task_preds, 
                                    const std::vector<std::vector<double>> price_diff, 
                                    double* sharpe_ratio, 
                                    double* average_holding_days, 
                                    double* strat_ret_vol, 
                                    double* annual_mean_strat_ret, 
                                    std::vector<double>* strategy_ret, 
                                    bool is_validate) {

    std::vector<double> top_50_correlations;
    std::vector<double> bottom_50_correlations;
    std::vector<double> rest_correlations;

    double total_return = 1.0;
    int holding_days = 0;
    int total_holding_periods = 0;
    double trading_days = 252.0; // Approximate trading days in a year
    int num_stocks = 50;
    double peak_value = 1.0; // Initialize peak value to the starting value of the portfolio
    double max_drawdown = 0.0; // Initialize max drawdown to zero    
    std::vector<int> prev_positions(all_task_preds.size(), 0); // 1 for long, -1 for short, 0 for no position

    for (size_t t = 0; t < all_task_preds[0].size(); ++t) {
        // std::cout << "Day " << t+1 << std::endl;  // Added debug print for each day
        double long_return = 0.0;
        double short_return = 0.0;
        int current_holding_days = 0;
        double transaction_cost_rate = 0.001;
        
        // Pair each stock with its prediction and sort by prediction
        std::vector<std::pair<double, int>> stock_preds;
        for (size_t n = 0; n < all_task_preds.size(); ++n) {
            stock_preds.push_back({all_task_preds[n][t], n});
        }
        std::sort(stock_preds.begin(), stock_preds.end(), std::greater<std::pair<double, int>>());

        // Buy top 50 stocks
        for (size_t n = 0; n < num_stocks; ++n) {
            int stock_index = stock_preds[n].second;
            long_return += price_diff[stock_index][t];
            // std::cout << "Bought stock " << stock_index << " with return: " << price_diff[stock_index][t] << std::endl;  // Debug print for stock bought
            if (prev_positions[stock_index] == 1) {
                current_holding_days++;
            } else {
                holding_days += current_holding_days;
                current_holding_days = 0;
                total_holding_periods++;
                prev_positions[stock_index] = 1;
                long_return -= transaction_cost_rate;
            }
        }

        // Sell bottom 50 stocks
        for (size_t n = all_task_preds.size() - num_stocks; n < all_task_preds.size(); ++n) {
            int stock_index = stock_preds[n].second;
            short_return -= price_diff[stock_index][t];
            // std::cout << "Sold stock " << stock_index << " with return: " << price_diff[stock_index][t] << std::endl;  // Debug print for stock sold
            if (prev_positions[stock_index] == -1) {
                current_holding_days++;
            } else {
                holding_days += current_holding_days;
                current_holding_days = 0;
                total_holding_periods++;
                prev_positions[stock_index] = -1;
                short_return -= transaction_cost_rate;
            }
        }

        // std::cout << "long_return " << long_return << std::endl;
        // std::cout << "short_return " << short_return << std::endl;
        double daily_return = (long_return + short_return) / (2 * num_stocks); 
        if (abs(daily_return) > 1) {continue;}
        total_return *= (1 + daily_return);
        strategy_ret->push_back(daily_return);

        // Collect predictions and true returns for top 50, bottom 50, and rest
        std::vector<double> top_50_preds, top_50_truths;
        std::vector<double> bottom_50_preds, bottom_50_truths;
        std::vector<double> rest_preds, rest_truths;

        for (size_t n = 0; n < num_stocks; ++n) {
            int stock_index = stock_preds[n].second;
            top_50_preds.push_back(all_task_preds[stock_index][t]);
            top_50_truths.push_back(price_diff[stock_index][t]);
        }
        for (size_t n = all_task_preds.size() - num_stocks; n < all_task_preds.size(); ++n) {
            int stock_index = stock_preds[n].second;
            bottom_50_preds.push_back(all_task_preds[stock_index][t]);
            bottom_50_truths.push_back(price_diff[stock_index][t]);
        }
        for (size_t n = num_stocks; n < all_task_preds.size() - num_stocks; ++n) {
            int stock_index = stock_preds[n].second;
            rest_preds.push_back(all_task_preds[stock_index][t]);
            rest_truths.push_back(price_diff[stock_index][t]);
        }

        // Calculate daily correlations and store
        top_50_correlations.push_back(CorrelationVec(top_50_preds, top_50_truths));
        bottom_50_correlations.push_back(CorrelationVec(bottom_50_preds, bottom_50_truths));
        rest_correlations.push_back(CorrelationVec(rest_preds, rest_truths));

        // Update peak value and calculate drawdown
        if (total_return > peak_value) {
            peak_value = total_return;
        } else {
            double drawdown = (peak_value - total_return) / peak_value;
            if (drawdown > max_drawdown) {
                max_drawdown = drawdown;
            }
        }        
    }

    // Calculate average correlations over time steps
    double avg_top_50_corr = std::accumulate(top_50_correlations.begin(), top_50_correlations.end(), 0.0) / top_50_correlations.size();
    double avg_bottom_50_corr = std::accumulate(bottom_50_correlations.begin(), bottom_50_correlations.end(), 0.0) / bottom_50_correlations.size();
    double avg_rest_corr = std::accumulate(rest_correlations.begin(), rest_correlations.end(), 0.0) / rest_correlations.size();

    // std::cout << "Average Pearson Correlation (Top 50): " << avg_top_50_corr << std::endl;
    // std::cout << "Average Pearson Correlation (Bottom 50): " << avg_bottom_50_corr << std::endl;
    // std::cout << "Average Pearson Correlation (Rest): " << avg_rest_corr << std::endl;

    // for (const auto &val : *strategy_ret) {
    //     std::cout << val << " ";
    // }
    // std::cout << std::endl;

    double mean_return = std::accumulate(strategy_ret->begin(), strategy_ret->end(), 0.0) / strategy_ret->size();

    *strat_ret_vol = stdev(*strategy_ret);
    *annual_mean_strat_ret = mean_return * trading_days;
    *sharpe_ratio = (*annual_mean_strat_ret) / (*strat_ret_vol * std::sqrt(trading_days));
    *average_holding_days = (double)holding_days / total_holding_periods;
    
    // Printing the metrics
    // std::cout << "Sharpe Ratio: " << *sharpe_ratio << std::endl;
    // std::cout << "Strategy Return Volatility: " << *strat_ret_vol << std::endl;
    // std::cout << "Annual Mean Strategy Return: " << *annual_mean_strat_ret << std::endl;
    // std::cout << "Average Holding Days: " << *average_holding_days << std::endl;
    // std::cout << "Total Portfolio Return: " << total_return - 1 << std::endl;
    // std::cout << "Maximum Drawdown: " << max_drawdown * 100 << "%" << std::endl; // Convert to percentage
    // exit(0);
    return max_drawdown;
}


double Evaluator::CorrelationVec(const std::vector<double> all_task_preds, const std::vector<double> price_diff) {

  IntegerT length = all_task_preds.size();
  if (all_task_preds.size() != price_diff.size()) length = std::min(all_task_preds.size(), price_diff.size());

  vector<double>::const_iterator first_all_task_preds = all_task_preds.begin() + (all_task_preds.size() - length);
  vector<double>::const_iterator last_all_task_preds = all_task_preds.end();
  vector<double> new_all_task_preds(first_all_task_preds, last_all_task_preds);

  vector<double>::const_iterator first_price_diff = price_diff.begin() + (price_diff.size() - length);
  vector<double>::const_iterator last_price_diff = price_diff.end();
  vector<double> new_price_diff(first_price_diff, last_price_diff);

  double sum = 0;
  for (std::vector<double>::size_type j = 0; j < new_all_task_preds.size(); j++) {
    sum += ((new_all_task_preds[j] - mean(new_all_task_preds)) * (new_price_diff[j] - mean(new_price_diff)));
  }

  sum = sum / (stdev(new_all_task_preds) * stdev(new_price_diff) * static_cast<double>(new_all_task_preds.size()));          

  return sum;
}

double Evaluator::Correlation(const std::vector<std::vector<double> > all_task_preds, const std::vector<std::vector<double> > price_diff) {

    CHECK(all_task_preds.size() == price_diff.size());
    for (std::vector<double>::size_type j = 0; j < all_task_preds.size(); j++) {
      CHECK(all_task_preds[j].size() == price_diff[j].size());
    }
    std::vector<double> preds_column;
    std::vector<double> price_diff_column;
    double result = 0;
    for (std::vector<double>::size_type i = 0; i < all_task_preds[0].size(); i++) 
      {      
        preds_column.clear();
        price_diff_column.clear();
        CHECK(preds_column.size() == 0);
        CHECK(price_diff_column.size() == 0);
        for (std::vector<double>::size_type j = 0; j < all_task_preds.size(); j++) {
          if (std::abs(all_task_preds[j][i]) != 1234 && std::abs(price_diff[j][i]) != 1234) {
            preds_column.push_back(all_task_preds[j][i]);
            price_diff_column.push_back(price_diff[j][i]);
          } 
        }
        CHECK(preds_column.size() == price_diff_column.size());
        double sum = 0;
        for (std::vector<double>::size_type j = 0; j < preds_column.size(); j++) {
          sum += ((preds_column[j] - mean(preds_column)) * (price_diff_column[j] - mean(price_diff_column)));
        }
        if (stdev(preds_column) == 0.0 || preds_column.size() == 0 || price_diff_column.size() == 0) { // after adding fec check, sometimes preds_column could have very few stocks' results even nothing. Adding this the latter two conditions check then can avoid preds_column equal to nothing. Add the first condition to check if any nonsense predictions.
          return 0; // if at one time step all preds are the same then must be not learning anything useful then we should return nothing at all;
        } else {
          result += sum / (stdev(preds_column) * stdev(price_diff_column) * static_cast<double>(preds_column.size()));          
        }
      }
      result = result/static_cast<double>(all_task_preds[0].size());

    return result;
}

double Evaluator::mean(const std::vector<double> v) {
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / v.size();
  return mean;
}

double Evaluator::stdev(const std::vector<double> v) {
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / v.size();

  std::vector<double> diff(v.size());
  std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / v.size());
  return stdev;
}

double Evaluator::Execute(const TaskInterface& task,
                          const IntegerT num_train_examples,
                          const Algorithm& algorithm,
                          std::vector<std::vector<double>> *all_task_preds,
                          std::vector<std::vector<double>> *all_price_diff,
                          std::vector<std::vector<std::vector<double>>>* tasks_rank,
                          IntegerT this_round,
                          IntegerT task_index,
                          IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, const IntegerT all_rounds, std::vector<IntegerT> *useful_list) {
  switch (task.FeaturesSize()) {
    case 2: {
      const Task<2>& downcasted_task = *SafeDowncast<2>(&task); // downcast is to change the task/dataset and then in execute task<F> is treated as dataset
      return ExecuteImpl<2>(downcasted_task, num_train_examples, algorithm, all_task_preds, all_price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
    }
    case 4: {
      const Task<4>& downcasted_task = *SafeDowncast<4>(&task);
      return ExecuteImpl<4>(downcasted_task, num_train_examples, algorithm, all_task_preds, all_price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
    }
    case 8: {
      const Task<8>& downcasted_task = *SafeDowncast<8>(&task);
      return ExecuteImpl<8>(downcasted_task, num_train_examples, algorithm, all_task_preds, all_price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
    }
    case 10: {
      const Task<10>& downcasted_task = *SafeDowncast<10>(&task);
      return ExecuteImpl<10>(downcasted_task, num_train_examples, algorithm, all_task_preds, all_price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
    }
    case 13: {
      const Task<13>& downcasted_task = *SafeDowncast<13>(&task);
      return ExecuteImpl<13>(downcasted_task, num_train_examples, algorithm, all_task_preds, all_price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
    }
    case 16: {
      const Task<16>& downcasted_task = *SafeDowncast<16>(&task);
      return ExecuteImpl<16>(downcasted_task, num_train_examples, algorithm, all_task_preds, all_price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
    }
    case 32: {
      const Task<32>& downcasted_task = *SafeDowncast<32>(&task);
      return ExecuteImpl<32>(downcasted_task, num_train_examples, algorithm,  all_task_preds, all_price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, all_rounds, useful_list);
    }
    default:
      LOG(FATAL) << "Unsupported features size." << endl;
  }
}

IntegerT Evaluator::GetNumTrainStepsCompleted() const {
  return num_train_steps_completed_;
}

template <FeatureIndexT F>
double Evaluator::ExecuteImpl(const Task<F>& task,
                              const IntegerT num_train_examples,
                              const Algorithm& algorithm,
                              std::vector<std::vector<double>>* all_task_preds,
                              std::vector<std::vector<double>>* all_price_diff,
                              std::vector<std::vector<std::vector<double>>>* tasks_rank,
                              IntegerT this_round,
                              IntegerT task_index,
                              IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, const IntegerT all_rounds, std::vector<IntegerT> *useful_list) {

  Executor<F> executor(
      algorithm, task, num_train_examples, task.ValidSteps(),
      rand_gen_, max_abs_error_);
  vector<double> valid_preds;
  vector<double> price_diff;
  const double fitness = executor.Execute(&valid_preds, &price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, nullptr, nullptr, useful_list);
  if (this_round == all_rounds - 1) {
    all_task_preds->push_back(valid_preds);
    all_price_diff->push_back(price_diff);
  }
  num_train_steps_completed_ += executor.GetNumTrainStepsCompleted();
  return fitness;
  
}

namespace internal {

double Median(vector<double> values) {  // Intentional copy.
  const size_t half_num_values = values.size() / 2;
  nth_element(values.begin(), values.begin() + half_num_values, values.end());
  return values[half_num_values];
}

double CombineFitnesses(
    const vector<double>& task_fitnesses,
    const FitnessCombinationMode mode) {
  if (mode == MEAN_FITNESS_COMBINATION) {
    double combined_fitness = 0.0;
    for (const double fitness : task_fitnesses) {
      combined_fitness += fitness;
    }
    combined_fitness /= static_cast<double>(task_fitnesses.size());
    return combined_fitness;
  } else if (mode == MEDIAN_FITNESS_COMBINATION) {
    return Median(task_fitnesses);
  } else {
    LOG(FATAL) << "Unsupported fitness combination." << endl;
  }
}

}  // namespace internal

}  // namespace alphaevolve
