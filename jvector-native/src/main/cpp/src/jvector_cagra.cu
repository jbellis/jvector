/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <mutex>

#include <cstdint>
#include <cstdlib>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_resources_manager.hpp>
#include <raft/random/make_blobs.cuh>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cuvs/neighbors/cagra.hpp>

struct jv_cagra_builder_t {
    raft::device_resources dev_resources;
    std::vector<std::vector<float>> host_vectors;
    int64_t dim;

    jv_cagra_builder_t(int32_t n_nodes, int64_t dim)
        : dev_resources(), dim(dim) {
        host_vectors.reserve(n_nodes);
    }
};

struct jv_cagra_index_t {
    cuvs::neighbors::cagra::index<float, uint32_t> index;

    jv_cagra_index_t(cuvs::neighbors::cagra::index<float, uint32_t>&& index)
        : index(std::move(index)) {}
};

extern "C" {
    jv_cagra_builder_t* create_cagra_builder(int32_t n_nodes, int64_t dim) {
        return new jv_cagra_builder_t(n_nodes, dim);
    }

    void add_node(jv_cagra_builder_t* builder, float* vector) {
        if (builder == nullptr || vector == nullptr) {
            return;
        }
        std::vector<float> vec(builder->dim);
        std::copy(vector, vector + builder->dim, vec.begin());
        builder->host_vectors.push_back(std::move(vec));
    }

    jv_cagra_index_t* build_cagra_index(jv_cagra_builder_t* builder) {
        if (builder == nullptr) {
            return nullptr;
        }

        using namespace cuvs::neighbors;
        int64_t n_nodes = builder->host_vectors.size();
        int64_t dim = builder->dim;

        // Create device matrices
        auto device_vectors = raft::make_device_matrix<float, int64_t>(builder->dev_resources, n_nodes, dim);

        // Copy data to device
        std::vector<float> flattened_vectors;
        for (const auto& vec : builder->host_vectors) {
            flattened_vectors.insert(flattened_vectors.end(), vec.begin(), vec.end());
        }
        raft::copy(device_vectors.data_handle(), flattened_vectors.data(), flattened_vectors.size(), raft::resource::get_cuda_stream(builder->dev_resources));
        builder->dev_resources.sync_stream();

        // Build the index
        // DEMOFIXME: tune index_params instead of using defaults
        cagra::index_params index_params;
        auto index = cagra::build(builder->dev_resources, index_params, raft::make_const_mdspan(device_vectors.view()));
        builder->dev_resources.sync_stream();
        return new jv_cagra_index_t(std::move(index));

        // DEMOFIXME: clean up builder resources
    }

    int32_t* search_cagra_index(jv_cagra_index_t* index, float* query, int32_t topk) {
        if (index == nullptr || query == nullptr) {
            return nullptr;
        }

        raft::device_resources const& res = raft::device_resources_manager::get_device_resources();

        using namespace cuvs::neighbors;

        // Prepare output arrays
        auto neighbors = raft::make_device_matrix<uint32_t>(res, 1, topk);
        auto distances = raft::make_device_matrix<float>(res, 1, topk);

        // Create an mdspan from the raw pointer
        auto span = raft::make_device_matrix_view<float, int64_t>(query, 1, index->index.dim());

        // Perform the search
        cagra::search_params search_params;
        search_params.itopk_size = topk;
        cagra::search(res, search_params, index->index, raft::make_const_mdspan(span), neighbors.view(), distances.view());

        // Copy results back to host
        std::vector<uint32_t> host_neighbors(topk);
        raft::copy(host_neighbors.data(), neighbors.data_handle(), topk, res.get_stream());
        raft::resource::sync_stream(res, res.get_stream());

        // DEMOFIXME: eliminate a copy
        // Allocate result array and copy the data
        int32_t* result = static_cast<int32_t*>(malloc(topk * sizeof(int32_t)));
        std::copy(host_neighbors.begin(), host_neighbors.end(), result);

        return result;
    }

    void free_cagra_index(jv_cagra_index_t* index) {
        if (index == nullptr) {
            return;
        }
        delete index;
    }

    void save_cagra_index(jv_cagra_index_t* index, const char* filename) {
        if (index == nullptr || filename == nullptr) {
            return;
        }

        raft::device_resources const& res = raft::device_resources_manager::get_device_resources();

        try {
            cuvs::neighbors::cagra::serialize(res, filename, index->index);
        } catch (const std::exception& e) {
            // Handle or log the error
            std::cerr << "Error saving CAGRA index: " << e.what() << std::endl;
        }
    }

    jv_cagra_index_t* load_cagra_index(const char* filename) {
        if (filename == nullptr) {
            return nullptr;
        }

        raft::device_resources const& res = raft::device_resources_manager::get_device_resources();

        try {
            // Create an index with default metric (L2Expanded)
            auto loaded_index = std::make_unique<cuvs::neighbors::cagra::index<float, uint32_t>>(res);

            // Deserialize into the created index
            cuvs::neighbors::cagra::deserialize(res, filename, loaded_index.get());

            // Create and return a new jv_cagra_index_t with the loaded index
            return new jv_cagra_index_t(std::move(*loaded_index));
        } catch (const std::exception& e) {
            // Handle or log the error
            std::cerr << "Error loading CAGRA index: " << e.what() << std::endl;
            return nullptr;
        }
    }
    }
