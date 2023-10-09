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

#ifndef EXPERIMENT_UTIL_H_
#define EXPERIMENT_UTIL_H_

#include <unordered_map>
#include <unordered_set>

#include "task.pb.h"
#include "definitions.h"
#include "instruction.pb.h"
#include "google/protobuf/repeated_field.h"

namespace alphaevolve {

std::vector<Op> ExtractOps(const google::protobuf::RepeatedField<int>& ops_src);

}  // namespace alphaevolve

#endif  // EXPERIMENT_UTIL_H_
