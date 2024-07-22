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

# Use RAPIDS_VERSION from cmake/thirdparty/fetch_rapids.cmake
set(CUVS_VERSION "${RAPIDS_VERSION}")
set(CUVS_FORK "rapidsai")
set(CUVS_PINNED_TAG "branch-${RAPIDS_VERSION}")

function(find_and_configure_cuvs)
    set(oneValueArgs VERSION FORK PINNED_TAG ENABLE_NVTX BUILD_CUVS_C_LIBRARY)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )


    set(CUVS_COMPONENTS "")
    if(PKG_BUILD_CUVS_C_LIBRARY)
        string(APPEND CUVS_COMPONENTS " c_api")
    endif()
    #-----------------------------------------------------
    # Invoke CPM find_package()
    #-----------------------------------------------------
    rapids_cpm_find(cuvs ${PKG_VERSION}
            GLOBAL_TARGETS      cuvs::cuvs
            BUILD_EXPORT_SET    cuvs-examples-exports
            INSTALL_EXPORT_SET  cuvs-examples-exports
            COMPONENTS ${CUVS_COMPONENTS}
            CPM_ARGS
            GIT_REPOSITORY https://github.com/${PKG_FORK}/cuvs.git
            GIT_TAG        ${PKG_PINNED_TAG}
            SOURCE_SUBDIR  cpp
            OPTIONS
            "BUILD_C_LIBRARY ${PKG_BUILD_CUVS_C_LIBRARY}"
            "BUILD_TESTS OFF"
            "CUVS_NVTX ${PKG_ENABLE_NVTX}"
            )
endfunction()

# Change pinned tag here to test a commit in CI
# To use a different CUVS locally, set the CMake variable
# CPM_cuvs_SOURCE=/path/to/local/cuvs
find_and_configure_cuvs(VERSION  ${CUVS_VERSION}.00
        FORK                     ${CUVS_FORK}
        PINNED_TAG               ${CUVS_PINNED_TAG}
        ENABLE_NVTX              OFF
        BUILD_CUVS_C_LIBRARY     ${BUILD_CUVS_C_LIBRARY}
)
