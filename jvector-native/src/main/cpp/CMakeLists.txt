#
# All changes to the original code are Copyright DataStax, Inc.
#
# Please see the included license file for details.
#


# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(cmake/thirdparty/fetch_rapids.cmake)

# Include necessary RAPIDS CMake modules
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(test_cuvs)

project(test_cuvs LANGUAGES CXX CUDA)

# ------------- configure local RAFT -----------------#

# Specify the path to the RAFT library
set(RAFT_LIBRARY_PATH "/home/jonathan/miniforge3/envs/cuvs/lib/libraft.so")

# Check if the library exists
if(NOT EXISTS ${RAFT_LIBRARY_PATH})
    message(FATAL_ERROR "RAFT library not found at ${RAFT_LIBRARY_PATH}")
endif()

# Create an imported target for RAFT
add_library(RAFT::RAFT SHARED IMPORTED)
set_target_properties(RAFT::RAFT PROPERTIES
    IMPORTED_LOCATION ${RAFT_LIBRARY_PATH}
)

# Specify the include directory for RAFT headers
# Adjust this path if necessary
set(RAFT_INCLUDE_DIR "/home/jonathan/miniforge3/envs/cuvs/include")

if(NOT EXISTS ${RAFT_INCLUDE_DIR})
    message(FATAL_ERROR "RAFT include directory not found at ${RAFT_INCLUDE_DIR}")
endif()

target_include_directories(RAFT::RAFT INTERFACE ${RAFT_INCLUDE_DIR})

# You may need to add dependencies for RAFT here, such as CUDA
find_package(CUDA REQUIRED)
target_link_libraries(RAFT::RAFT INTERFACE ${CUDA_LIBRARIES})

# ------------- configure cuvs -----------------#

rapids_cpm_init()
set(BUILD_CUVS_C_LIBRARY OFF)

# Find CUVS package
find_package(cuvs REQUIRED)

# -------------- compile tasks ----------------- #
add_library(jvectorgpu SHARED src/jvector_gpupq.cu src/cagra_example.cu src/jvector_cagra.cu)
target_link_libraries(jvectorgpu PRIVATE
    cuvs::cuvs
    RAFT::RAFT
    $<TARGET_NAME_IF_EXISTS:conda_env>
)
