#!/bin/bash

#
# All changes to the original code are Copyright DataStax, Inc.
#
# Please see the included license file for details.
#

# Copyright (c) 2023-2024, NVIDIA CORPORATION.

# raft empty project template build script

# Abort script on first error
set -e

PARALLEL_LEVEL=${PARALLEL_LEVEL:=`nproc`}

BUILD_TYPE=Release
BUILD_DIR=build/

RAFT_REPO_REL=""
EXTRA_CMAKE_ARGS=""
set -e


if [[ ${RAFT_REPO_REL} != "" ]]; then
  RAFT_REPO_PATH="`readlink -f \"${RAFT_REPO_REL}\"`"
  EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCPM_raft_SOURCE=${RAFT_REPO_PATH}"
fi

if [ "$1" == "clean" ]; then
  rm -rf build
  exit 0
fi

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake \
 -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
 -DRAFT_NVTX=OFF \
 -DCMAKE_CUDA_ARCHITECTURES="NATIVE" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 ${EXTRA_CMAKE_ARGS} \
 ../

cmake  --build . -j${PARALLEL_LEVEL}

cp libjvectorgpu.so ../../resources/

# Generate Java source code
# Should only be run when c header changes
# Check if jextract is available before running
if ! command -v jextract &> /dev/null
then
    echo "WARNING: jextract could not be found, please install it if you need to update bindings."
    exit 0
fi

jextract \
  --output ../../java \
  -t io.github.jbellis.jvector.vector.cnative \
  -I . \
  --header-class-name NativeGpuOps \
  ../src/jvector_gpupq.h

# Set critical linker option with heap-based segments for all generated methods
sed -i 's/DESC)/DESC, Linker.Option.critical(true))/g' ../../java/io/github/jbellis/jvector/vector/cnative/NativeGpuOps.java
