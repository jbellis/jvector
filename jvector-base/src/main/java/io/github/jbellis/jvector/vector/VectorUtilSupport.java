/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.List;

/**
 * Interface for implementations of VectorUtil support.
 */
public interface VectorUtilSupport {

  /** Calculates the dot product of the given float arrays. */
  float dotProduct(VectorFloat<?> a, VectorFloat<?> b);

  /** Calculates the dot product of float arrays of differing sizes, or a subset of the data */
  float dotProduct(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length);

  /** Returns the cosine similarity between the two vectors. */
  float cosine(VectorFloat<?> v1, VectorFloat<?> v2);

  /** Calculates the cosine similarity of VectorFloats of differing sizes, or a subset of the data */
  float cosine(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length);

  /** Returns the sum of squared differences of the two vectors. */
  float squareDistance(VectorFloat<?> a, VectorFloat<?> b);

  /** Calculates the sum of squared differences of float arrays of differing sizes, or a subset of the data */
  float squareDistance(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length);

  /** returns the sum of the given vectors. */
  VectorFloat<?> sum(List<VectorFloat<?>> vectors);

  /** return the sum of the components of the vector */
  float sum(VectorFloat<?> vector);

  /** Divide vector by divisor, in place (vector will be modified) */
  void scale(VectorFloat<?> vector, float multiplier);

  /** Adds v2 into v1, in place (v1 will be modified) */
  void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2);

  /** Subtracts v2 from v1, in place (v1 will be modified) */
  void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2);

  /** @return a - b, element-wise */
  VectorFloat<?> sub(VectorFloat<?> a, VectorFloat<?> b);

  /** @return a - b, element-wise, starting at aOffset and bOffset respectively */
  VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length);

  /**
   * Calculates the sum of sparse points in a vector.
   * <p>
   * This assumes the data vector is a 2d matrix which has been flattened into 1 dimension
   * so rather than data[n][m] it's data[n * m].  With this layout this method can quickly
   * assemble the data from this heap and sum it.
   *
   * @param data the vector of all datapoints
   * @param baseIndex the start of the data in the offset table
   *                  (scaled by the index of the lookup table)
   * @param baseOffsets bytes that represent offsets from the baseIndex
   * @return the sum of the points
   */
  float assembleAndSum(VectorFloat<?> data, int baseIndex, ByteSequence<?> baseOffsets);

  int hammingDistance(long[] v1, long[] v2);

  /**
   * Calculates the similarity score of multiple product quantization-encoded vectors against a single query vector,
   * using precomputed similarity score fragments derived from codebook contents.
   * @param shuffles a sequence of shuffles to be used against partial pre-computed fragments. These are transposed PQ-encoded
   *                 vectors using the same codebooks as the partials. Due to the transposition, rather than this being
   *                 contiguous encoded vectors, the first component of all vectors is stored contiguously, then the second, and so on.
   * @param codebookCount The number of codebooks used in the PQ encoding.
   * @param partials The precomputed score fragments for each codebook entry. These are stored as a contiguous vector of all
   *                 the fragments for one codebook, followed by all the fragments for the next codebook, and so on.
   * @param vsf      The similarity function to use.
   * @param results  The output vector to store the similarity scores. This should be pre-allocated to the same size as the number of shuffles.
   */
  void bulkShuffleSimilarity(ByteSequence<?> shuffles, int codebookCount, VectorFloat<?> partials, VectorSimilarityFunction vsf, VectorFloat<?> results);

  void calculatePartialSums(VectorFloat<?> codebook, int baseOffset, int size, int clusterCount, VectorFloat<?> query, int offset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums);

  default void dotProductMultiScore(VectorFloat<?> v1, VectorFloat<?> v2, VectorFloat<?> results) {
    for (int i = 0; i < results.length(); i++) {
      results.set(i, (1 + dotProduct(v1, 0, v2, i * v1.length(), v1.length())) / 2);
    }
  }

  default void squareL2DistanceMultiScore(VectorFloat<?> v1, VectorFloat<?> v2, VectorFloat<?> results) {
    for (int i = 0; i < results.length(); i++) {
      results.set(i, 1 / (1 + squareDistance(v1, 0, v2, i * v1.length(), v1.length())));
    }
  }

  default void cosineMultiScore(VectorFloat<?> v1, VectorFloat<?> v2, VectorFloat<?> results) {
    for (int i = 0; i < results.length(); i++) {
      results.set(i, (1 + cosine(v1, 0, v2, i * v1.length(), v1.length())) / 2);
    }
  }
}
