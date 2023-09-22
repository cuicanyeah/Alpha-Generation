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

#ifndef TASK_UTIL_H_
#define TASK_UTIL_H_

#include <array>
#include <fstream>
#include <random>
#include <type_traits>

#include "task.h"
#include "task.pb.h"
#include "definitions.h"
#include "executor.h"
#include "generator.h"
#include "memory.h"
#include "random_generator.h"
#include "absl/strings/str_cat.h"

namespace alphaevolve {

using ::std::shuffle;  // NOLINT
using ::std::vector;  // NOLINT
using ::std::cout;  // NOLINT

const RandomSeedT kFirstParamSeedForTest = 9000;
const RandomSeedT kFirstDataSeedForTest = 19000;

// Fills `return_tasks` with `experiment_tasks` Tasks.
// `return_tasks` must be empty an empty vector.
void FillTasks(
    const TaskCollection& task_collection,
    std::vector<std::unique_ptr<TaskInterface>>* return_tasks);

// Downcasts a TaskInterface. Crashes if the downcast would have been
// incorrect.
template <FeatureIndexT F>
Task<F>* SafeDowncast(TaskInterface* task) {
  CHECK(task != nullptr);
  CHECK_EQ(task->FeaturesSize(), F);
  return dynamic_cast<Task<F>*>(task);
}
template <FeatureIndexT F>
const Task<F>* SafeDowncast(const TaskInterface* task) {
  CHECK(task != nullptr);
  CHECK_EQ(task->FeaturesSize(), F);
  return dynamic_cast<const Task<F>*>(task);
}
template <FeatureIndexT F>
std::unique_ptr<Task<F>> SafeDowncast(
    std::unique_ptr<TaskInterface> task) {
  CHECK(task != nullptr);
  return std::unique_ptr<Task<F>>(SafeDowncast<F>(task.release()));
}

namespace test_only {

// Convenience method to generates a single task from a string containing a
// text-format TaskSpec. Only generates 1 task, so `num_tasks` can be
// 1 or can be left unset. The features size is determined by the template
// argument `F`, so the `features_size` in the TaskSpec can be equal to `F`
// or be left unset.
template <FeatureIndexT F>
Task<F> GenerateTask(const std::string& task_spec_str) {
  TaskCollection task_collection;
  TaskSpec* task_spec = task_collection.add_tasks();
  CHECK(google::protobuf::TextFormat::ParseFromString(task_spec_str, task_spec));
  if (!task_spec->has_features_size()) {
    task_spec->set_features_size(F);
  }
  CHECK_EQ(task_spec->features_size(), F);
  if (!task_spec->has_num_tasks()) {
    task_spec->set_num_tasks(1);
  }
  CHECK_EQ(task_spec->num_tasks(), 1);
  std::vector<std::unique_ptr<TaskInterface>> tasks;
  FillTasks(task_collection, &tasks);
  CHECK_EQ(tasks.size(), 1);

  std::unique_ptr<Task<F>> task = SafeDowncast<F>(std::move(tasks[0]));
  return std::move(*task);
}

}  // namespace test_only

// james: add a bool as the flag used in below codes to decide whether make valid buffer large to output alpha data as feature
// bool whether_output_alpha_data = false;

template <FeatureIndexT F>
struct ClearAndResizeImpl {
  static void Call(const IntegerT num_train_examples,
                   const IntegerT num_valid_examples,
                   TaskBuffer<F>* buffer) {
    buffer->train_features_.clear();
    buffer->train_labels_.clear();
    buffer->valid_features_.clear();
    buffer->valid_labels_.clear();
    buffer->train_features_.resize(num_train_examples-(F-1));
    buffer->train_labels_.resize(num_train_examples-(F-1));
    if (false) {
      buffer->valid_features_.resize(num_valid_examples-(F-1)+num_train_examples-(F-1));
      buffer->valid_labels_.resize(num_valid_examples-(F-1)+num_train_examples-(F-1));      
    }
    else {
      buffer->valid_features_.resize(num_valid_examples-(F-1));
      buffer->valid_labels_.resize(num_valid_examples-(F-1));      
    }
    for (Matrix<F>& value : buffer->train_features_) {
      value.resize(F, F);
    }
    for (Matrix<F>& value : buffer->valid_features_) {
      value.resize(F, F);
    }
  }
};

template <FeatureIndexT F>
void ClearAndResize(const IntegerT num_train_examples,
                    const IntegerT num_valid_examples,
                    TaskBuffer<F>* buffer) {
  ClearAndResizeImpl<F>::Call(num_train_examples, num_valid_examples, buffer);
}

template <FeatureIndexT F>
struct StockTaskCreator {
  static void Create(EvalType eval_type,
                     const StockTask& task_spec,
                     IntegerT num_train_examples, IntegerT num_valid_examples,
                     IntegerT features_size, RandomSeedT data_seed, IntegerT ith_task,
                     TaskBuffer<F>* buffer) {
    ClearAndResize(num_train_examples, num_valid_examples, buffer);

    std::string path;
    CHECK(task_spec.has_path() & !task_spec.path().empty())
        << "You have to specifiy the path to the data!" << std::endl;
    path = task_spec.path();

    IntegerT positive_class;
    IntegerT negative_class;

    if (task_spec.has_positive_class() &&
        task_spec.has_negative_class()) {
      positive_class = task_spec.positive_class();
      negative_class = task_spec.negative_class();
      IntegerT num_supported_data_seeds =
          task_spec.max_supported_data_seed() -
          task_spec.min_supported_data_seed();
      data_seed = static_cast<RandomSeedT>(
          task_spec.min_supported_data_seed() +
          data_seed % num_supported_data_seeds);
    } else if (!task_spec.has_positive_class() &&
               !task_spec.has_negative_class()) {
      std::mt19937 task_bit_gen(HashMix(
          static_cast<RandomSeedT>(856572777), data_seed));
      RandomGenerator task_gen(&task_bit_gen);      

      data_seed = static_cast<RandomSeedT>(
          task_gen.UniformInteger(
              task_spec.min_supported_data_seed(),
              task_spec.max_supported_data_seed())); 
      // Generate the key using the task_spec.
      std::string filename = absl::StrCat(
          "dataset_", task_spec.dataset_name(), "-stock_",
          ith_task,
          "-dim_", features_size, "-seed_", data_seed);

      std::string full_path = path + "/" + filename;
      ScalarLabelDataset saved_dataset;
      std::ifstream is(full_path, std::ifstream::binary);
      CHECK(is.good()) << "No data found at " << full_path
          << (". Please generate "
              "the stock datasets first.") << std::endl;
      if (is.good()) {
        std::string read_buffer((std::istreambuf_iterator<char>(is)),
                                std::istreambuf_iterator<char>());
        CHECK(saved_dataset.ParseFromString(read_buffer))
            << "Error while parsing the proto from "
            << full_path << std::endl;
        is.close();
      }

      /// pass the fourteenth feature to industry relation iterator
      buffer->industry_relation_ = saved_dataset.train_features(0).features(F);

      // /// random check the fourteenth feature ind relation is correct
      // if (saved_dataset.train_features(0).features(F) != saved_dataset.train_features(100).features(F)) {
      //   for (IntegerT i_dim = 0; i_dim < F + 1; ++i_dim) {
      //     cout << "saved_dataset.train_features(100).features(F)" << saved_dataset.train_features(100).features(i_dim) << endl;
      //   }
      // }
      // CHECK_EQ(saved_dataset.train_features(0).features(F), saved_dataset.train_features(100).features(F));
      // CHECK_EQ(saved_dataset.valid_features(0).features(F), saved_dataset.valid_features(100).features(F));

      if (saved_dataset.train_features(0).features(F) == -1234) cout << "saved_dataset.train_features(0).features(F)" << saved_dataset.train_features(0).features(F) << endl;

      // Check there is enough data saved in the sstable.
      // std::cout << "saved_dataset.valid_features_size()" << saved_dataset.valid_features_size() << std::endl;

      CHECK_GE(saved_dataset.train_features_size(),
               buffer->train_features_.size())
          << "Not enough training examples in " << full_path << std::endl;
      CHECK_GE(saved_dataset.train_labels_size(),
               buffer->train_labels_.size())
           << "Not enough training labels in " << full_path << std::endl;
      // std::cout << "buffer->train_features_.size()" << buffer->train_features_.size() << std::endl;

      for (IntegerT k = F-1; k < buffer->train_features_.size()+(F-1); ++k)  {
        for (IntegerT j_dim = 0; j_dim < F; ++j_dim) {
          for (IntegerT i_dim = 0; i_dim < F; ++i_dim) {
             buffer->train_features_[k-(F-1)](i_dim, j_dim) =
                 saved_dataset.train_features(k-(F-1-j_dim)).features(i_dim);
           }
        }
        if (saved_dataset.train_labels(k-(F-1)) > 1) cout << "saved_dataset.train_labels(k) > 1!!!!!!!!!" << saved_dataset.train_labels(k) << endl;

        buffer->train_labels_[k-(F-1)] = saved_dataset.train_labels(k);
      }

      // james: add flag here to decide whether make valid buffer large to output alpha data as feature
      if (false) {
        CHECK_GE(saved_dataset.valid_features_size()+2*saved_dataset.train_features_size(),
                 buffer->valid_features_.size());
        CHECK_GE(saved_dataset.valid_labels_size()+2*saved_dataset.train_features_size(),
                 buffer->valid_labels_.size());
        CHECK_EQ(F, 13);

        for (IntegerT k = F-1; k < buffer->train_features_.size()+(F-1); ++k)  {
          for (IntegerT j_dim = 0; j_dim < F; ++j_dim) {
            for (IntegerT i_dim = 0; i_dim < F; ++i_dim) {
               buffer->valid_features_[k-(F-1)](i_dim, j_dim) =
                   saved_dataset.train_features(k-(F-1-j_dim)).features(i_dim);
             }
          }
          if (saved_dataset.train_labels(k-(F-1)) > 1) cout << "saved_dataset.train_labels(k) > 1!!!!!!!!!" << saved_dataset.train_labels(k) << endl;

          buffer->valid_labels_[k-(F-1)] = saved_dataset.train_labels(k);
        }
        // std::cout << "buffer->valid_features_.size()" << buffer->valid_features_.size() << std::endl;
        for (IntegerT k = F-1+buffer->train_features_.size(); k < buffer->valid_features_.size()+(F-1); ++k) {
          for (IntegerT j_dim = 0; j_dim < F; ++j_dim) {
            for (IntegerT i_dim = 0; i_dim < F; ++i_dim) {
               buffer->valid_features_[k-(F-1)](i_dim, j_dim) =
                   saved_dataset.valid_features(k-(F-1-j_dim)-buffer->train_features_.size()).features(i_dim);
             }
          }

          buffer->valid_labels_[k-(F-1)] = saved_dataset.valid_labels(k-buffer->train_features_.size());
        } 
      }
      else {
        CHECK_GE(saved_dataset.valid_features_size(),
                 buffer->valid_features_.size());
        CHECK_GE(saved_dataset.valid_labels_size(),
                 buffer->valid_labels_.size());
        CHECK_EQ(F, 13);
        // std::cout << "buffer->valid_features_.size()" << buffer->valid_features_.size() << std::endl;
        for (IntegerT k = F-1; k < buffer->valid_features_.size()+(F-1); ++k)  {
          for (IntegerT j_dim = 0; j_dim < F; ++j_dim) {
            for (IntegerT i_dim = 0; i_dim < F; ++i_dim) {
               buffer->valid_features_[k-(F-1)](i_dim, j_dim) =
                   saved_dataset.valid_features(k-(F-1-j_dim)).features(i_dim);
             }
          }

          buffer->valid_labels_[k-(F-1)] = saved_dataset.valid_labels(k);
        }        
      }
      std::string filename_price_diff = "/hdd7/james/HSI_AlphaEvolve/debug_can_delete_why_dont_match/long_data/" +std::to_string(ith_task)+ "_labelstrue.txt";  
      std::ofstream outFilelabelst(filename_price_diff);
      for (IntegerT k = 0; k < buffer->valid_features_.size(); ++k) {
        outFilelabelst << buffer->valid_labels_[k] << " ";
      }      
      outFilelabelst.close();  
    } else {
      LOG(FATAL) << ("You should either provide both or none of the positive"
                     " and negative classes.") << std::endl;
    }
  }
};

// Creates a task using the linear regressor with fixed weights. The
// weights are determined by the seed. Serves as a way to initialize the
// task.
template <FeatureIndexT F>
struct ScalarLinearRegressionTaskCreator {
  static void Create(EvalType eval_type, IntegerT num_train_examples,
                     IntegerT num_valid_examples, RandomSeedT param_seed,
                     RandomSeedT data_seed, TaskBuffer<F>* buffer) {
    ClearAndResize(num_train_examples, num_valid_examples, buffer);
    std::mt19937 data_bit_gen(data_seed + 939723201);
    RandomGenerator data_gen(&data_bit_gen);
    std::mt19937 param_bit_gen(param_seed + 997958712);
    RandomGenerator weights_gen(&param_bit_gen);
    Generator generator(NO_OP_ALGORITHM, 0, 0, 0, {}, {}, {}, nullptr,
                        nullptr);

    // Create a Algorithm and memory deterministically.
    Algorithm algorithm = generator.LinearModel(0.0);
    Memory<F> memory;
    memory.Wipe();
    weights_gen.FillGaussian<F>(
        0.0, 1.0, &memory.vector_[Generator::LINEAR_ALGORITHMWeightsAddress]);

    // Fill in the labels by executing the Algorithm.
    ExecuteAndFillLabels<F>(algorithm, &memory, buffer, &data_gen);
    CHECK_EQ(eval_type, RMS_ERROR);
  }
};

// Creates a task using the nonlinear regressor with fixed weights. The
// weights are determined by the seed. Serves as a way to initialize the
// task.
template <FeatureIndexT F>
struct Scalar2LayerNnRegressionTaskCreator {
  static void Create(EvalType eval_type, IntegerT num_train_examples,
                     IntegerT num_valid_examples, RandomSeedT param_seed,
                     RandomSeedT data_seed, TaskBuffer<F>* buffer) {
    ClearAndResize(num_train_examples, num_valid_examples, buffer);
    std::mt19937 data_bit_gen(data_seed + 865546086);
    RandomGenerator data_gen(&data_bit_gen);
    std::mt19937 param_bit_gen(param_seed + 174299604);
    RandomGenerator weights_gen(&param_bit_gen);
    Generator generator(NO_OP_ALGORITHM, 0, 0, 0, {}, {}, {}, nullptr,
                        nullptr);

    // Create a Algorithm and memory deterministically.
    Algorithm algorithm = generator.UnitTestNeuralNetNoBiasNoGradient(0.0);
    Memory<F> memory;
    memory.Wipe();
    weights_gen.FillGaussian<F>(
        0.0, 1.0, &memory.matrix_[
        Generator::kUnitTestNeuralNetNoBiasNoGradientFirstLayerWeightsAddress]);
    for (FeatureIndexT col = 0; col < F; ++col) {
      memory.matrix_[
          Generator::kUnitTestNeuralNetNoBiasNoGradientFirstLayerWeightsAddress]
          (0, col) = 0.0;
      memory.matrix_[
          Generator::kUnitTestNeuralNetNoBiasNoGradientFirstLayerWeightsAddress]
          (2, col) = 0.0;
    }
    weights_gen.FillGaussian<F>(
        0.0, 1.0,
        &memory.vector_[
        Generator::kUnitTestNeuralNetNoBiasNoGradientFinalLayerWeightsAddress]);
    memory.vector_[
        Generator::kUnitTestNeuralNetNoBiasNoGradientFinalLayerWeightsAddress]
        (0) = 0.0;
    memory.vector_[
        Generator::kUnitTestNeuralNetNoBiasNoGradientFinalLayerWeightsAddress]
        (2) = 0.0;

    // Fill in the labels by executing the Algorithm.
    ExecuteAndFillLabels<F>(algorithm, &memory, buffer, &data_gen);
    CHECK_EQ(eval_type, RMS_ERROR);
  }
};

template<FeatureIndexT F>
void CopyUnitTestFixedTaskVector(
    const google::protobuf::RepeatedField<double>& src, Scalar* dest) {
  LOG(FATAL) << "Not allowed." << std::endl;
}
template<FeatureIndexT F>
void CopyUnitTestFixedTaskVector(
    const google::protobuf::RepeatedField<double>& src, Vector<F>* dest) {
  CHECK_EQ(src.size(), F);
  for (IntegerT index = 0; index < F; ++index) {
    (*dest)(index) = src.at(index);
  }
}

template <FeatureIndexT F>
struct UnitTestFixedTaskCreator {
  static void Create(const UnitTestFixedTask& task_spec,
                     TaskBuffer<F>* buffer) {
    const IntegerT num_train_examples = task_spec.train_features_size();
    CHECK_EQ(task_spec.train_labels_size(), num_train_examples);
    const IntegerT num_valid_examples = task_spec.valid_features_size();
    CHECK_EQ(task_spec.valid_labels_size(), num_valid_examples);
    ClearAndResize(num_train_examples, num_valid_examples, buffer);
  }
};

template <FeatureIndexT F>
struct UnitTestZerosTaskCreator {
  static void Create(const IntegerT num_train_examples,
                     const IntegerT num_valid_examples,
                     const UnitTestZerosTaskSpec& task_spec,
                     TaskBuffer<F>* buffer) {
    ClearAndResize(num_train_examples, num_valid_examples, buffer);
    for (Matrix<F>& feature : buffer->train_features_) {
      feature.setZero();
    }
    for (Scalar& label : buffer->train_labels_) {
      label = 0.0;
    }
    for (Matrix<F>& feature : buffer->valid_features_) {
      feature.setZero();
    }
    for (Scalar& label : buffer->valid_labels_) {
      label = 0.0;
    }
  }
};

template <FeatureIndexT F>
struct UnitTestOnesTaskCreator {
  static void Create(const IntegerT num_train_examples,
                     const IntegerT num_valid_examples,
                     const UnitTestOnesTaskSpec& task_spec,
                     TaskBuffer<F>* buffer) {
    ClearAndResize(num_train_examples, num_valid_examples, buffer);
    for (Matrix<F>& feature : buffer->train_features_) {
      feature.setOnes();
    }
    for (Scalar& label : buffer->train_labels_) {
      label = 1.0;
    }
    for (Matrix<F>& feature : buffer->valid_features_) {
      feature.setOnes();
    }
    for (Scalar& label : buffer->valid_labels_) {
      label = 1.0;
    }
  }
};

template <FeatureIndexT F>
struct UnitTestIncrementTaskCreator {
  static void Create(const IntegerT num_train_examples,
                     const IntegerT num_valid_examples,
                     const UnitTestIncrementTaskSpec& task_spec,
                     TaskBuffer<F>* buffer) {
    ClearAndResize(num_train_examples, num_valid_examples, buffer);

    const double increment = task_spec.increment();
    Scalar incrementing_scalar = 0.0;
    Matrix<F> incrementing_vector = Matrix<F>::Zero(F, F);
    Matrix<F> ones_vector = Matrix<F>::Ones(F, F);
    ones_vector *= increment;

    for (Matrix<F>& feature : buffer->train_features_) {
      feature = incrementing_vector;
      incrementing_vector += ones_vector;
    }
    for (Scalar& label : buffer->train_labels_) {
      label = incrementing_scalar;
      incrementing_scalar += increment;
    }

    incrementing_scalar = 0.0;
    incrementing_vector.setZero();

    for (Matrix<F>& feature : buffer->valid_features_) {
      feature = incrementing_vector;
      incrementing_vector += ones_vector;
    }
    for (Scalar& label : buffer->valid_labels_) {
      label = incrementing_scalar;
      incrementing_scalar += increment;
    }
  }
};

template <FeatureIndexT F>
std::unique_ptr<Task<F>> CreateTask(const IntegerT task_index,
                                    const RandomSeedT param_seed,
                                    const RandomSeedT data_seed,
                                    const TaskSpec& task_spec,
                                    const IntegerT ith_task) {
  CHECK_GT(task_spec.num_train_examples(), 0);
  CHECK_GT(task_spec.num_valid_examples(), 0);
  TaskBuffer<F> buffer;
  switch (task_spec.task_type_case()) {
    case (TaskSpec::kStockTask):
      StockTaskCreator<F>::Create(
          task_spec.eval_type(),
          task_spec.stock_task(),
          task_spec.num_train_examples(), task_spec.num_valid_examples(),
          task_spec.features_size(), data_seed, ith_task, &buffer);
      break;
    case (TaskSpec::kScalarLinearRegressionTask):
      ScalarLinearRegressionTaskCreator<F>::Create(
          task_spec.eval_type(), task_spec.num_train_examples(),
          task_spec.num_valid_examples(), param_seed, data_seed, &buffer);
      break;
    case (TaskSpec::kScalar2LayerNnRegressionTask):
      Scalar2LayerNnRegressionTaskCreator<F>::Create(
          task_spec.eval_type(), task_spec.num_train_examples(),
          task_spec.num_valid_examples(), param_seed, data_seed, &buffer);
      break;
    case (TaskSpec::kUnitTestFixedTask):
      UnitTestFixedTaskCreator<F>::Create(
          task_spec.unit_test_fixed_task(), &buffer);
      break;
    case (TaskSpec::kUnitTestZerosTask):
      UnitTestZerosTaskCreator<F>::Create(
          task_spec.num_train_examples(),
          task_spec.num_valid_examples(),
          task_spec.unit_test_zeros_task(),
          &buffer);
      break;
    case (TaskSpec::kUnitTestOnesTask):
      UnitTestOnesTaskCreator<F>::Create(
          task_spec.num_train_examples(),
          task_spec.num_valid_examples(),
          task_spec.unit_test_ones_task(),
          &buffer);
      break;
    case (TaskSpec::kUnitTestIncrementTask):
      UnitTestIncrementTaskCreator<F>::Create(
          task_spec.num_train_examples(), task_spec.num_valid_examples(),
          task_spec.unit_test_increment_task(), &buffer);
      break;
    default:
      LOG(FATAL) << "Unknown task type\n";
      break;
  }

  std::mt19937 data_bit_gen(data_seed + 3274582109);

  CHECK(task_spec.has_eval_type());
  return absl::make_unique<Task<F>>(
      task_index, task_spec.eval_type(),
      task_spec.num_train_epochs(), &data_bit_gen, &buffer);
}

// Randomizes all the seeds given a base seed. See "internal workflow" comment
// in task.proto.
void RandomizeTaskSeeds(TaskCollection* task_collection,
                        RandomSeedT seed);

}  // namespace alphaevolve

#endif  // TASK_UTIL_H_
