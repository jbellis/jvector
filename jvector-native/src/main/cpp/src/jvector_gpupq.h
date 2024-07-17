/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdint.h>

#ifndef JVECTOR_GPUPQ_H
#define JVECTOR_GPUPQ_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jpq_dataset_t jpq_dataset_t;
typedef struct jpq_query_t jpq_query_t;

// Function to load jpq_dataset
jpq_dataset_t* load_pq_vectors(const char* filename);

// Function to load query
jpq_query_t* load_query(jpq_dataset_t* dataset, const float* query);

// Function to compute L2 similarities
void compute_dp_similarities(jpq_query_t* query_handle, const int32_t* node_ids, float* similarities, int64_t n_nodes);

// Function to free jpq_dataset
// DEMOFIXME: this needs an implementation
void free_jpq_dataset(jpq_dataset_t* dataset);

// Function to free jpq_query
// DEMOFIXME: this needs an implementation, to free the lookup table
void free_jpq_query(jpq_query_t* query_handle);

void run_jpq_test_cohere(void);

void initialize(void);

#ifdef __cplusplus
}
#endif

#endif
