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
typedef struct jpq_adc_t jpq_adc_t;
typedef struct jpq_query_t jpq_query_t;
typedef struct jv_cagra_builder_t jv_cagra_builder_t;
typedef struct jv_cagra_index_t jv_cagra_index_t;

// Function to load jpq_dataset
jpq_dataset_t* load_pq_vectors(const char* filename);

// Function to free jpq_dataset
void free_jpq_dataset(jpq_dataset_t* dataset);

// Function to prepare adc query
jpq_adc_t* prepare_adc_query(jpq_dataset_t* dataset, const float* queries, int32_t n_queries);

// Function to compute dot product similarities with adc
void compute_dp_similarities_adc(jpq_adc_t* adc_handle, const int32_t* node_ids, float* similarities, int32_t nodes_per_query);

// Function to free adc query
void free_adc_query(jpq_adc_t* query_handle);

// Function to prepare query
jpq_query_t* prepare_query(jpq_dataset_t* dataset, const float* query);

// Function to free query
void free_query(jpq_query_t* query_handle);

// Function to compute dot product similarities
void compute_dp_similarities(jpq_query_t* query_handle, const int32_t* node_ids, float* similarities, int64_t n_nodes);

void run_jpq_test_cohere(void);

void initialize(void);

float* allocate_float_vector(int32_t length);

int32_t* allocate_node_ids(int32_t length);

jv_cagra_builder_t* create_cagra_builder(int32_t n_nodes, int64_t dim);

void add_node(jv_cagra_builder_t* builder, float* vector);

jv_cagra_index_t* build_cagra_index(jv_cagra_builder_t* builder);

int32_t* search_cagra_index(jv_cagra_index_t* index, float* query, int32_t topk);

void free_cagra_index(jv_cagra_index_t* index);

void save_cagra_index(jv_cagra_index_t* index, const char* filename);

jv_cagra_index_t* load_cagra_index(const char* filename);

int call_cagra_demo(void);

#ifdef __cplusplus
}
#endif

#endif
